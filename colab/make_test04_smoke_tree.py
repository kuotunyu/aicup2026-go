"""Drive stand-in tree for the CPU smoke run of colab/test04_rank.ipynb, and the checks after that run
(results/test04/design/spec.md 8.5 L2b). CPU torch only; the scratch folder must be outside the repository.

    PY colab/make_test04_smoke_tree.py --scratch S [--threads 6] [--rerun]    1. build the stand-in tree
    PY colab/run_local_smoke.py colab/test04_rank.ipynb --scratch S           2. run the notebook (C10_SMOKE=1)
    PY colab/make_test04_smoke_tree.py --scratch S --check [--threads 6]      3. check the run

Set OMP_NUM_THREADS / MKL_NUM_THREADS to the same value as --threads for step 2 (the shared machine: <= 6).

1. build writes
   S/drive/C10_bundle_v1.zip        byte copy of data/colab_bundle/C10_bundle_v1.zip (run_local_smoke.py copies it too)
   S/drive/results/prereg03_smoke/rank_{a,b}/ckpt/final.pth
                                    a random GoRankResNet(in_channels=501) state_dict: torch.manual_seed(1) for arm a,
                                    torch.manual_seed(2) for arm b
   S/drive/results/prereg03_smoke/rank_{a,b}/dumps/{val_keys,val_p03x_d4,val_p03x_s0_swap}.npz + _DONE_ markers
                                    the seam reference, computed by the prereg03 notebook's own code (the view_d4 cell, the
                                    three official-14 functions and the dump-tools functions of colab/prereg03_rank_a.ipynb,
                                    executed here with multiprocessing.dummy for Pool) on the first 12 validation
                                    questions (SMOKE_CFG['questions']), exactly as prereg03's SMOKE run writes them
   S/drive/C10_test_v1.zip          kind smoke, built from training players (never from validation games):
                                    tools/t02_fake_tests.py from-train --shape official --slot-colours filled,
                                    tools/t02_checks.py t1 --dry-run --only T1.1,T1.2,T1.3 (rank, player),
                                    tools/t02_test_bundle.py build --kind smoke (tree S/fake_tests)
   S/drive/results/test02_smoke/rank/{config,run_log}.json
                                    stand-in records of a completed test02 smoke run on that test bundle (marked
                                    "stand_in"), so the notebook's preflight step 5 compares something
   S/SMOKE_TREE_T04.json            what was built (SHA-256 of every file, seeds, commands, seconds)
3. --check reads S/drive/results/test04_smoke and writes S/L2B_CHECK.json:
   * run_log status completed, seam_check passed, both arms bitwise equal to the reference (d4 and swap), zip members;
   * seam_val: the notebook's five npz files are byte-identical to the stand-in reference;
   * the QUESTIONS_PATH adapter: for public and private, the prereg03 notebook's own dump_val_p03 (executed from
     colab/prereg03_rank_a.ipynb with RANK_QUESTIONS = that set's question file in S/work/tests) gives npz files
     byte-identical to the test04 notebook's four logits files and val_keys.npz of that set, and
     config.dumps.<set>.questions_path is that question file.
Exit code 0 when everything holds, 1 otherwise (or on a refusal).
"""

import argparse
import ast
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from multiprocessing.dummy import Pool as ThreadPool

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BUNDLE_SRC = os.path.join(REPO, "data", "colab_bundle", "C10_bundle_v1.zip")
BUNDLE_NAME = "C10_bundle_v1"
P03_NB = os.path.join(REPO, "colab", "prereg03_rank_a.ipynb")
P03_NB_SHA256 = "85e0c2ae3269c68ad166310efb2525b2d48ed1279c9cf6eb6b428e59e8693557"
T04_NB = os.path.join(REPO, "colab", "test04_rank.ipynb")
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
ARMS = ("a", "b")
SEEDS = {"a": 1, "b": 2}
DUMP_NAMES = {"a": "p03a", "b": "p03b"}
SETS = ("seam_val", "public", "private")
TEST_FILES = {"public": "rank_prediction_test_public.csv", "private": "rank_prediction_test_private.csv"}
NEEDLE = "ans" + "wers"   # validation label files are never opened (spelled in halves on purpose)


class SmokeTreeError(Exception):
    pass


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def inside(path, root):
    p, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def refuse(path):
    if NEEDLE in os.path.basename(path).lower():
        raise SmokeTreeError("refusing a validation label file: %s" % path)
    return path


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------- the prereg03 notebook's own code
def _base_key(meta):
    if "new" in meta:
        return meta["new"]
    b = meta["base"]
    return "official_%d" % b["official_cell"] if "official_cell" in b else b["new"]


def p03_cells():
    with open(P03_NB, "rb") as f:
        raw = f.read()
    if sha_bytes(raw) != P03_NB_SHA256:
        raise SmokeTreeError("colab/prereg03_rank_a.ipynb is not the committed one (%s)" % sha_bytes(raw)[:12])
    nb = json.loads(raw.decode("utf-8"))
    out = {}
    for c in nb["cells"]:
        m = c["metadata"]["c10"]
        out[_base_key(m["base"]) if "base_cell" in m else m["new"]] = "".join(c["source"])
    return out


def exec_nodes(src, ns, defs=(), assigns=()):
    """Execute only the named top-level function definitions and single-name assignments of a cell."""
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in defs)
            or (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in assigns)]
    got = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in keep}
    if got != set(defs) | set(assigns):
        raise SmokeTreeError("cell lacks %s" % sorted(set(defs) | set(assigns) - got))
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<prereg03 cell>", "exec"), ns)


def p03_namespace(questions_path, tutorial_dir, quiet=True):
    """A namespace holding prereg03's dump code as its SMOKE run has it (CPU, 12 questions, threads for Pool)."""
    import numpy as np
    import pandas as pd
    import torch
    if tutorial_dir not in sys.path:
        sys.path.insert(0, tutorial_dir)
    from network import GoRankResNet
    from utils import SGFParseRankPrediction
    cells = p03_cells()
    ns = {"__name__": "prereg03_smoke_reference", "np": np, "pd": pd, "torch": torch, "os": os, "io": io,
          "json": json, "time": time, "math": math, "hashlib": hashlib, "GoRankResNet": GoRankResNet,
          "SGFParseRankPrediction": SGFParseRankPrediction, "Pool": ThreadPool, "SMOKE": True,
          "DEVICE": torch.device("cpu"), "IN_CHANNELS": 501, "NUM_CLASSES": len(LEVELS), "LEVELS": list(LEVELS),
          "RUN": {"run": 1},
          # the notebook's environment cell on a CPU-only torch: no GPU, no cuDNN
          "ENV_INFO": {"gpu": None,
                       "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None},
          "READONLY_ROOTS": {}, "save_config": lambda **kw: None, "mark_time": lambda *a, **k: None,
          "RANK_QUESTIONS": refuse(questions_path), "DUMP_DIR": None}
    if quiet:
        ns["print"] = lambda *a, **k: None
    import datetime
    ns["datetime"] = datetime
    exec_nodes(cells["setup"], ns, defs=("now_iso", "_real", "_is_under", "guard_write", "write_json", "read_json",
                                         "sha256_path", "gpu_name_of"), assigns=("SMOKE_CFG",))
    ns["NPROC"] = ns["SMOKE_CFG"]["nproc"]
    exec(compile(cells["view_d4"], "<prereg03 view_d4>", "exec"), ns)   # runs the cell's own unit checks too
    exec_nodes(cells["official_14"], ns, defs=("state_dict_sha256", "_nchw_output", "channels_last"))
    exec_nodes(cells["dump_tools"], ns,
               defs=("load_net", "_dump_path", "_marker_path", "dump_is_done", "write_dump", "require_gpu_for",
                     "write_keys", "sgf_to_features", "parse_val", "dump_val_p03"),
               assigns=("DUMP_MAX_BYTES", "GPU_BATCH", "VAL", "_infer_parser"))
    return ns


def p03_dump(ns, dump_dir, weights_path):
    """prereg03's key file and final dump (key_files cell: write_keys('val_keys', parse_val()['keys']); final_dumps
    cell: load_net(final.pth) then dump_val_p03(net, 'p03' + arm, 'prereg03/rank_<arm>/ckpt/final.pth', sha))."""
    arm = os.path.basename(os.path.dirname(os.path.dirname(weights_path)))[len("rank_"):]
    os.makedirs(dump_dir, exist_ok=True)
    ns["DUMP_DIR"] = dump_dir
    keys = ns["write_keys"]("val_keys", ns["parse_val"]()["keys"])
    net, wsha = ns["load_net"](weights_path)
    status = ns["dump_val_p03"](net, DUMP_NAMES[arm], "prereg03/rank_%s/ckpt/final.pth" % arm, wsha)
    return {"keys": keys, "dump": status, "weights_sha256": wsha}


# ---------------------------------------------------------------------------- build
def run_tool(args, log):
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("PYTHONUTF8", None)
    r = subprocess.run([sys.executable] + args, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    log.append({"command": " ".join(args), "exit": r.returncode, "seconds": round(time.time() - t0, 1),
                "stdout_tail": r.stdout[-1500:], "stderr_tail": r.stderr[-1500:]})
    if r.returncode != 0:
        raise SmokeTreeError("%s exited %d:\n%s\n%s" % (" ".join(args), r.returncode, r.stdout[-3000:], r.stderr[-3000:]))
    return r


def extract_bundle(ref_work):
    """The bundle members the reference needs (official utils/network, the validation question file), checked
    against the bundle manifest."""
    with zipfile.ZipFile(BUNDLE_SRC) as z:
        manifest = json.loads(z.read(BUNDLE_NAME + "/bundle_manifest.json"))
        for rel in ("tutorial/utils.py", "tutorial/network.py", "local/rank_val_questions.csv"):
            data = z.read("%s/%s" % (BUNDLE_NAME, rel))
            if sha_bytes(data) != manifest["files"][rel]:
                raise SmokeTreeError("bundle member %s does not match its manifest" % rel)
            dst = os.path.join(ref_work, BUNDLE_NAME, *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(data)
    return manifest


def cmd_build(a):
    import torch
    S = a.scratch
    drive, fake, ref_work = os.path.join(S, "drive"), os.path.join(S, "fake_tests"), os.path.join(S, "ref_work")
    p03_root = os.path.join(drive, "results", "prereg03_smoke")
    targets = [drive, fake, ref_work, os.path.join(S, "work"), os.path.join(S, "adapter_check")]
    present = [p for p in targets if os.path.exists(p)]
    if present and not a.rerun:
        raise SmokeTreeError("already there (use --rerun to start over): %s" % present)
    for p in present:
        shutil.rmtree(p)
    t_start = time.time()
    log, record = [], {"format": "c10-test04 smoke tree v1", "what": __doc__.split("\n")[0], "scratch": S,
                       "threads": a.threads, "torch": torch.__version__, "python": sys.version.split()[0]}
    torch.set_num_threads(a.threads)

    os.makedirs(drive)
    shutil.copyfile(BUNDLE_SRC, os.path.join(drive, os.path.basename(BUNDLE_SRC)))
    manifest = extract_bundle(ref_work)
    tutorial = os.path.join(ref_work, BUNDLE_NAME, "tutorial")
    qval = os.path.join(ref_work, BUNDLE_NAME, "local", "rank_val_questions.csv")
    record["bundle"] = {"zip_sha256": sha_file(BUNDLE_SRC), "manifest_files": manifest["files"]}

    # random weights of the two arms (the stand-in for prereg03's final.pth)
    sys.path.insert(0, tutorial)
    from network import GoRankResNet
    record["weights"] = {}
    for x in ARMS:
        path = os.path.join(p03_root, "rank_" + x, "ckpt", "final.pth")
        os.makedirs(os.path.dirname(path))
        torch.manual_seed(SEEDS[x])
        net = GoRankResNet(in_channels=501, num_classes=len(LEVELS))
        torch.save(net.state_dict(), path)
        record["weights"][x] = {"path": os.path.relpath(path, S), "seed": SEEDS[x], "sha256": sha_file(path),
                                "bytes": os.path.getsize(path)}

    # the seam reference: prereg03's own code on the first 12 validation questions
    t0 = time.time()
    ns = p03_namespace(qval, tutorial)
    record["reference"] = {}
    for x in ARMS:
        d = os.path.join(p03_root, "rank_" + x, "dumps")
        st = p03_dump(ns, d, os.path.join(p03_root, "rank_" + x, "ckpt", "final.pth"))
        record["reference"][x] = {"status": st, "files": {n: sha_file(os.path.join(d, n)) for n in sorted(os.listdir(d))}}
    record["reference_seconds"] = round(time.time() - t0, 1)

    # the smoke test bundle, from training players only
    run_tool(["tools/t02_fake_tests.py", "from-train", "--root", fake, "--shape", "official", "--slot-colours",
              "filled"], log)
    for comp in ("rank", "player"):
        run_tool(["tools/t02_checks.py", "t1", "--comp", comp, "--dry-run", "--root", fake, "--only",
                  "T1.1,T1.2,T1.3"], log)
    built = os.path.join(fake, "built", "data", "test")
    run_tool(["tools/t02_test_bundle.py", "build", "--kind", "smoke", "--root", fake, "--tests-dir",
              os.path.join(fake, "data", "test"), "--out-dir", built], log)
    test_zip = os.path.join(fake, "data", "colab_bundle", "C10_test_v1.zip")
    shutil.copyfile(test_zip, os.path.join(drive, "C10_test_v1.zip"))
    with zipfile.ZipFile(test_zip) as z:
        man_raw = z.read("tests/MANIFEST.json")
    man = json.loads(man_raw)
    if man.get("kind") != "smoke":
        raise SmokeTreeError("the built test bundle is not kind smoke: %r" % man.get("kind"))
    record["test_bundle"] = {"zip_sha256": sha_file(test_zip), "manifest_sha256": sha_bytes(man_raw),
                             "kind": man.get("kind"), "t1": man.get("t1"),
                             "files": {n: e["sha256"] for n, e in man["files"].items()}}

    # stand-in test02 smoke records on the same test bundle (preflight step 5 reads them, read only)
    t02_rank = os.path.join(drive, "results", "test02_smoke", "rank")
    stand_in = "written by colab/make_test04_smoke_tree.py for the test04 CPU smoke; not a test02 run"
    write_json(os.path.join(t02_rank, "config.json"),
               {"stand_in": stand_in, "smoke": True, "rehearsal": False, "test_kind": "smoke",
                "test_bundle": record["test_bundle"], "status": "completed"})
    write_json(os.path.join(t02_rank, "run_log.json"),
               {"stand_in": stand_in, "status": "completed", "runs": [{"run": 1, "status": "completed"}]})
    record["test02_smoke_stand_in"] = {n: sha_file(os.path.join(t02_rank, n)) for n in ("config.json", "run_log.json")}
    record["commands"] = log
    record["seconds"] = round(time.time() - t_start, 1)
    record["next"] = "PY colab/run_local_smoke.py colab/test04_rank.ipynb --scratch %s" % S
    write_json(os.path.join(S, "SMOKE_TREE_T04.json"), record)
    print(json.dumps({k: record[k] for k in ("weights", "reference_seconds", "test_bundle", "seconds")},
                     ensure_ascii=False, indent=1))
    print("next:", record["next"])
    return 0


# ---------------------------------------------------------------------------- check
def npz_files(stem_names):
    return ["val_keys.npz"] + ["val_%s_%s.npz" % (DUMP_NAMES[x], v) for x in ARMS for v in ("d4", "s0_swap")] \
        if stem_names is None else stem_names


def cmd_check(a):
    import torch
    S = a.scratch
    torch.set_num_threads(a.threads)
    drive, work = os.path.join(S, "drive"), os.path.join(S, "work")
    root = os.path.join(drive, "results", "test04_smoke")
    sw = os.path.join(root, "sw")
    p03_root = os.path.join(drive, "results", "prereg03_smoke")
    tutorial = os.path.join(S, "ref_work", BUNDLE_NAME, "tutorial")
    problems, out = [], {"format": "c10-test04 L2b check v1", "scratch": S, "threads": a.threads}
    t_start = time.time()

    run_log, seam, cfg = (read_json(os.path.join(sw, n)) for n in ("run_log.json", "seam_check.json", "config.json"))
    out["run_log_status"] = run_log.get("status")
    out["seam"] = {k: seam.get(k) for k in ("passed", "tol", "bitwise_equal", "same_gpu", "keys_equal", "weights_ok",
                                            "env_same", "seam_dump_gpu", "reference_gpu")}
    out["seam"]["arms"] = {x: {k: seam["arms"][x].get(k) for k in (
        "passed", "c0_s0_max_abs_diff", "c0_swap_max_abs_diff", "bitwise_equal_d4", "bitwise_equal_swap",
        "file_sha_equal_d4", "file_sha_equal_swap", "max_abs_logit_diff_per_sym", "max_abs_logit_diff_swap",
        "max_abs_e_diff_s0", "max_abs_e_diff_swap", "rule_b_differences_s0_record_only")} for x in ARMS}
    if run_log.get("status") != "completed":
        problems.append("run_log status %r" % run_log.get("status"))
    if seam.get("passed") is not True or seam.get("bitwise_equal") is not True:
        problems.append("seam passed %r, bitwise_equal %r" % (seam.get("passed"), seam.get("bitwise_equal")))
    for x in ARMS:
        arm = seam["arms"][x]
        if not all(arm.get(k) is True for k in ("bitwise_equal_d4", "bitwise_equal_swap", "file_sha_equal_d4",
                                                  "file_sha_equal_swap")):
            problems.append("arm %s not bitwise equal to the reference" % x)
    if cfg.get("format") != "c10-test04 rank notebook config v1" or cfg.get("smoke") is not True:
        problems.append("config format/smoke: %r %r" % (cfg.get("format"), cfg.get("smoke")))
    out["config"] = {k: cfg.get(k) for k in ("format", "smoke", "rehearsal", "kind", "notebook_sha256", "status",
                                             "attempt", "test02_record", "tf32")}

    # the zip
    zpath = os.path.join(root, "test04_rank_results.zip")
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
    need = (["sw/" + n for n in ("config.json", "run_log.json", "env.json", "timings.json", "seam_check.json",
                                 "readonly_before.json", "readonly_after.json")]
            + ["sw/%s/dumps/%s" % (s, n) for s in SETS for n in npz_files(None) + ["_INPUT.json"]
               + ["_DONE_%s.json" % f[:-4] for f in npz_files(None)]])
    missing = sorted(set(need) - names)
    out["zip"] = {"path": os.path.relpath(zpath, S), "sha256": sha_file(zpath), "bytes": os.path.getsize(zpath),
                  "members": len(names), "required": len(need), "missing": missing,
                  "all_under_sw": all(n.startswith("sw/") for n in names)}
    if missing or not out["zip"]["all_under_sw"]:
        problems.append("zip members: missing %s" % missing[:5])

    # seam_val: byte copies of the reference
    out["seam_val_vs_reference"] = {}
    for x in ARMS:
        ref = os.path.join(p03_root, "rank_" + x, "dumps")
        for n in ("val_keys.npz", "val_%s_d4.npz" % DUMP_NAMES[x], "val_%s_s0_swap.npz" % DUMP_NAMES[x]):
            same = sha_file(os.path.join(sw, "seam_val", "dumps", n)) == sha_file(os.path.join(ref, n))
            out["seam_val_vs_reference"]["%s/%s" % (x, n)] = same
            if not same:
                problems.append("seam_val %s differs from the reference of arm %s" % (n, x))

    # the QUESTIONS_PATH adapter: prereg03's own dump_val_p03 on each test set's question file
    want_paths = {"seam_val": os.path.join(work, BUNDLE_NAME, "local", "rank_val_questions.csv"),
                  "public": os.path.join(work, "tests", TEST_FILES["public"]),
                  "private": os.path.join(work, "tests", TEST_FILES["private"])}
    out["questions_path"] = {}
    for s in SETS:
        got = ((cfg.get("dumps") or {}).get(s) or {}).get("questions_path")
        ok = got is not None and os.path.normcase(os.path.abspath(got)) == os.path.normcase(os.path.abspath(want_paths[s]))
        out["questions_path"][s] = {"config": got, "expected": want_paths[s], "equal": ok}
        if not ok:
            problems.append("config.dumps.%s.questions_path is %r, not %r" % (s, got, want_paths[s]))
    out["adapter"] = {}
    t0 = time.time()
    for s in ("public", "private"):
        ns = p03_namespace(want_paths[s], tutorial)
        d = os.path.join(S, "adapter_check", s)
        if os.path.exists(d):
            shutil.rmtree(d)
        st = {x: p03_dump(ns, d, os.path.join(p03_root, "rank_" + x, "ckpt", "final.pth")) for x in ARMS}
        cmp = {n: sha_file(os.path.join(d, n)) == sha_file(os.path.join(sw, s, "dumps", n)) for n in npz_files(None)}
        out["adapter"][s] = {"status": st, "byte_identical": cmp,
                             "rows": read_json(os.path.join(d, "_DONE_val_keys.json"))["rows"]}
        if not all(cmp.values()):
            problems.append("adapter %s: %s" % (s, [n for n, v in cmp.items() if not v]))
    out["adapter_seconds"] = round(time.time() - t0, 1)
    out["timings_notebook"] = read_json(os.path.join(sw, "timings.json"))["runs"][-1]["seconds"]
    out["problems"] = problems
    out["ok"] = not problems
    out["seconds"] = round(time.time() - t_start, 1)
    write_json(os.path.join(S, "L2B_CHECK.json"), out)
    print(json.dumps({k: out[k] for k in ("ok", "problems", "run_log_status", "seam", "zip", "seam_val_vs_reference",
                                          "questions_path", "adapter", "adapter_seconds", "seconds")},
                     ensure_ascii=False, indent=1))
    return 0 if out["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scratch", required=True, help="scratch folder outside the repository")
    ap.add_argument("--check", action="store_true", help="check a finished run_local_smoke.py run")
    ap.add_argument("--threads", type=int, default=6, help="torch threads (1-8; the machine is shared)")
    ap.add_argument("--rerun", action="store_true", help="(build) delete an earlier stand-in tree first")
    a = ap.parse_args(argv)
    a.scratch = os.path.abspath(a.scratch)
    try:
        if inside(a.scratch, REPO):
            raise SmokeTreeError("--scratch %s is inside the repository" % a.scratch)
        if not 1 <= a.threads <= 8:
            raise SmokeTreeError("--threads must be 1-8")
        import torch
        if torch.cuda.is_available():
            raise SmokeTreeError("CPU-only torch expected (the GPU is not used for smoke runs)")
        return cmd_check(a) if a.check else cmd_build(a)
    except SmokeTreeError as e:
        print("REFUSED:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
