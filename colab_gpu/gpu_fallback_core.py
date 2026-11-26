"""C10 competition 2: the registered Colab GPU fallback for the 07 HS and 08 P1 steps (docs/PREREG_11_GPU_FALLBACK.md).

This file is the whole program the Colab notebook colab_gpu/gpu_fallback.ipynb runs (the notebook only mounts Drive,
installs the pinned packages, unpacks the kit and the input zip and calls `python gpu_fallback_core.py run ...`). The
same file runs locally for the CPU smoke (--fake-scorer) and is imported by tools/g11_gpu_fallback.py for its chunk
writer, so the chunks of the local drills and of Colab come from one piece of code.

    python colab_gpu/gpu_fallback_core.py run --kit K --input I --out O [--stage hs|p1] [--set NAME] [--workers W]
                                              [--stop-after-min M] [--tmp DIR]
                                              [--fake-scorer --tools REPO/tools] [--max-games N]   (local drills only)
    python colab_gpu/gpu_fallback_core.py verify --kit K [--input I] [--no-ckpt-hash]

K     the kit root: the repository layout of the pinned files (third_party/katago, results/prereg07/explore/
      pilot_humansl/{hs_lib,pilot_run,pilot_fuse}.py, results/prereg08/p1/{p1_lib,p1_extract,p1_fuse}.py,
      results/prereg05/explore/u1/u1_lib.py, data/katago/b18c384nbt-humanv0.ckpt, colab_gpu/gpu_fallback_core.py) and
      KIT_MANIFEST.json (SHA-256 of every file, the registered constants). Locally K may be the repository itself.
I     the input root: INPUT_MANIFEST.json and sets/<set>.json (games: sgf_content, colour, ref; keys; key / question
      file SHA-256), built by tools/g11_gpu_fallback.py input.
O     the output root (on Colab a Drive folder): O/<stage>/<set>/chunk_<i>.npz, O/<stage>/ENV.json (the one
      environment of the stage), O/<stage>/<set>/RUN_LOG.jsonl, O/RUN_SUMMARY.json.

What a chunk holds (byte format of tools/t07_rank.py hs and tools/t08_rank.py p1; the same fields in the same order,
the same format strings): hs = pilot_run.run_chunk on 200 games, 14 profiles, target side, fp16, batch 1024,
channels_last, cudnn.benchmark; p1 = p1_extract.run_chunk with the neutral metadata, both sides. Every chunk records
the key / question file SHA-256 of its set, the full checkpoint SHA-256, the KataGo commit, hs_lib's SHA-256 (hs) or
the P1 code record (p1), torch / cuDNN / GPU / driver, created_utc and the mode of the input (real for the test day,
rehearsal for the validation rehearsal). tool_sha256 is this file's SHA-256 (CRLF folded to LF).

Resume: run the same command again. A chunk that exists is reused only when it is this run's (format, refs, keys,
questions, mode, settings, checkpoint, code; else exit 2) and of this environment (torch / cuDNN / GPU; else exit 4:
PREREG_11 allows a resume only on the same GPU, torch and cuDNN). O/<stage>/ENV.json fixes the environment of a stage
at its first chunk. STOP file in O/<stage>/<set>/ or --stop-after-min end the run after the chunk in progress (exit 5).
Exit codes: 0 every requested set complete; 2 refused; 4 environment differs from the stage's; 5 stopped on request.

Standard library + numpy at module level; torch, hs_lib / pilot_run / p1_lib / p1_extract only inside run (and only
from the kit, after their SHA-256 are checked). Nothing here reads a validation label file; the input holds question
content only.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

CORE_FORMAT = "c10-g11 colab gpu fallback v1"
KIT_FORMAT = "c10-g11 kit v1"
INPUT_FORMAT = "c10-g11 input v1"
ENV_FORMAT = "c10-g11 stage environment v1"
SUMMARY_FORMAT = "c10-g11 run summary v1"
STAGES = ("hs", "p1")
MODES = ("real", "rehearsal", "dry-run", "smoke")
EXIT_REFUSED, EXIT_DRIFT, EXIT_STOPPED = 2, 4, 5
CHUNK_SET_PREFIX = {"hs": "test07_", "p1": "test08_"}
# the kit paths (repository layout; hs_lib / p1_lib compute the repository root from their own location)
PILOT_DIR = ("results", "prereg07", "explore", "pilot_humansl")
P1_DIR = ("results", "prereg08", "p1")
CKPT_REL = "data/katago/b18c384nbt-humanv0.ckpt"
SELF_REL = "colab_gpu/gpu_fallback_core.py"


class Refused(RuntimeError):
    code = EXIT_REFUSED


class Drift(RuntimeError):
    code = EXIT_DRIFT


# ================================================================ small helpers
def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha_raw(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha_lf(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def read_json(path):
    if not os.path.exists(path):
        raise Refused(f"missing {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def log_line(path, **kv):
    rec = {"time_utc": now_iso(), **kv}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def kit_path(kit, rel):
    return os.path.join(kit, *rel.split("/"))


def chunk_plan(n, size):
    """[(chunk index, first game, end)] in 200-game steps (t07: range(0, n, C); t08: p08_p1.chunk_plan)."""
    return [(ci, b, min(n, b + size)) for ci, b in enumerate(range(0, n, size))]


def chunk_name(ci):
    return f"chunk_{ci:05d}.npz"


# ================================================================ the kit and the input
def load_kit_manifest(kit):
    m = read_json(os.path.join(kit, "KIT_MANIFEST.json"))
    if m.get("format") != KIT_FORMAT:
        raise Refused(f"KIT_MANIFEST.json format {m.get('format')!r} is not {KIT_FORMAT!r}")
    return m


def verify_kit(kit, manifest=None, ckpt_hash=True, log=None):
    """Every file of the kit manifest has its raw SHA-256 (and the pinned LF SHA-256 where the manifest lists one); the
    checkpoint is hashed in full. -> record."""
    m = manifest or load_kit_manifest(kit)
    bad = []
    for rel, rec in sorted(m["files"].items()):
        p = kit_path(kit, rel)
        if rel == CKPT_REL:
            continue
        if not os.path.exists(p):
            bad.append(f"{rel} missing")
        elif sha_raw(p) != rec["sha256"]:
            bad.append(f"{rel} raw SHA-256 differs")
        elif rec.get("sha256_lf") and sha_lf(p) != rec["sha256_lf"]:
            bad.append(f"{rel} LF SHA-256 differs")
    for rel, want in sorted((m.get("pins_lf") or {}).items()):
        p = kit_path(kit, rel)
        if not os.path.exists(p) or sha_lf(p) != want:
            bad.append(f"{rel} is not the pinned file")
    if bad:
        raise Refused(f"kit check failed: {bad[:8]}")
    ck = {"path": CKPT_REL, "checked": False}
    p = kit_path(kit, CKPT_REL)
    if not os.path.exists(p):
        raise Refused(f"{CKPT_REL} missing from the kit")
    ck["bytes"] = os.path.getsize(p)
    if ck["bytes"] != m["constants"]["ckpt_bytes"]:
        raise Refused(f"{CKPT_REL}: {ck['bytes']} bytes, registered {m['constants']['ckpt_bytes']}")
    if ckpt_hash:
        t0 = time.time()
        ck["sha256"] = sha_raw(p)
        ck["seconds"] = round(time.time() - t0, 1)
        ck["checked"] = True
        if ck["sha256"] != m["constants"]["ckpt_sha256"]:
            raise Refused(f"{CKPT_REL}: SHA-256 {ck['sha256']} is not the registered {m['constants']['ckpt_sha256']}")
    if log:
        log(f"kit OK: {len(m['files'])} files, checkpoint {'hashed' if ck['checked'] else 'size only'}")
    return {"files": len(m["files"]), "checkpoint": ck, "kit_id": m.get("kit_id")}


def load_input(inp, kit_manifest=None):
    """INPUT_MANIFEST.json and its set files (SHA-256 checked). -> manifest with 'sets_data'."""
    m = read_json(os.path.join(inp, "INPUT_MANIFEST.json"))
    if m.get("format") != INPUT_FORMAT:
        raise Refused(f"INPUT_MANIFEST.json format {m.get('format')!r} is not {INPUT_FORMAT!r}")
    if m.get("mode") not in MODES:
        raise Refused(f"input mode {m.get('mode')!r}")
    if kit_manifest is not None and m.get("kit_id") != kit_manifest.get("kit_id"):
        raise Refused(f"the input was built for kit {m.get('kit_id')!r}, this kit is {kit_manifest.get('kit_id')!r}")
    data = {}
    for name, rec in m["sets"].items():
        p = os.path.join(inp, *rec["file"].split("/"))
        if not os.path.exists(p) or sha_raw(p) != rec["sha256"]:
            raise Refused(f"input set {name}: {rec['file']} missing or not its SHA-256")
        d = read_json(p)
        if d.get("set") != name or len(d["games"]) != rec["games"] or d.get("keys_sha256") != rec["keys_sha256"] \
                or d.get("questions_sha256") != rec["questions_sha256"]:
            raise Refused(f"input set {name}: the set file differs from INPUT_MANIFEST.json")
        if [g["ref"] for g in d["games"]] != list(range(len(d["games"]))):
            raise Refused(f"input set {name}: refs are not 0..n-1 in order")
        data[name] = d
    for r in m["runs"]:
        if r["stage"] not in STAGES or r["set"] not in data:
            raise Refused(f"input run {r} is not a stage / set of this input")
    m["sets_data"] = data
    return m


# ================================================================ environment
def nvidia_query():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        rows = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        out = [dict(zip(("name", "driver"), [y.strip() for y in x.split(",")])) for x in rows]
        return {"gpus": out, "driver": out[0]["driver"] if out else None}
    except (OSError, subprocess.SubprocessError):
        return {"gpus": [], "driver": None}


def environment(fake):
    """The fields of p07_hs.gpu_env / p08_p1.gpu_env (python, torch, numpy, cuda, cudnn, gpu, driver) plus record-only
    platform details."""
    if fake:
        env = {"python": sys.version.split()[0], "torch": "fake", "cudnn": "fake", "gpu": "fake", "driver": "fake",
               "numpy": np.__version__}
    else:
        import torch
        env = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
               "cuda": torch.cuda.is_available(), "cudnn": torch.backends.cudnn.version(),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "driver": nvidia_query().get("driver")}
        env["record_only"] = {"cuda_runtime": torch.version.cuda, "platform": platform.platform(),
                              "cpu_count": os.cpu_count(), "nvidia_smi": nvidia_query().get("gpus")}
    return env


ENV_KEY_FIELDS = ("torch", "cudnn", "gpu", "python", "numpy")


def env_key(env):
    """The environment that must not change within a stage and that a rehearsal allows: torch, cuDNN, GPU (what t07 /
    t08 chunk_check compare) plus python and numpy (PREREG_11 3; the driver is recorded only: Colab updates it)."""
    return {k: str(env.get(k)) for k in ENV_KEY_FIELDS}


def allowed_problem(I, stage, env):
    """PREREG_11 3 / review major 1: a real (test-day) input carries the environments its committed rehearsals
    allow; this runtime must be one of them before anything is written (a refusal here is not an attempt).
    -> None or the refusal text. Rehearsal, dry-run and smoke inputs are exempt."""
    if I.get("mode") != "real":
        return None
    allowed = (I.get("allowed_environments") or {}).get(stage) or []
    if not allowed:
        return f"the real input holds no allowed environment for {stage} (rebuild the input after the rehearsal)"
    key = env_key(env)
    if key not in allowed:
        return (f"this runtime {key} is not an environment the rehearsal allowed for {stage} ({allowed}); pick the "
                f"allowed GPU (nothing was written; this is not an attempt)")
    return None


def stage_environment(out, stage, env, input_id, mode):
    """O/<stage>/ENV.json: written at the stage's first chunk; afterwards the environment must be the same (PREREG_11:
    resume only on the same GPU / torch / cuDNN). -> record (with run_id)."""
    p = os.path.join(out, stage, "ENV.json")
    if os.path.exists(p):
        rec = read_json(p)
        if rec.get("format") != ENV_FORMAT or rec.get("input_id") != input_id or rec.get("mode") != mode:
            raise Refused(f"{p} belongs to another input or mode ({rec.get('input_id')}, {rec.get('mode')})")
        if rec["key"] != env_key(env):
            raise Drift(f"this runtime ({env_key(env)}) differs from the stage's environment {rec['key']} "
                        f"(PREREG_11: resume only on the same GPU, torch and cuDNN; this is a new attempt, which "
                        f"the registration does not allow)")
        return rec
    rec = {"format": ENV_FORMAT, "stage": stage, "input_id": input_id, "mode": mode, "run_id": uuid.uuid4().hex,
           "created_utc": now_iso(), "key": env_key(env), "environment": env}
    write_json_atomic(p, rec)
    return rec


# ================================================================ chunk fields (the order of t07_rank / t08_rank)
def hs_fields(*, M, refs, set_label, env, ck, keys_sha, questions_sha, tool_sha, mode, created_utc, seconds, res):
    """tools/t07_rank.py hs_run's np.savez keyword order, field for field."""
    c = M["constants"]
    run = c["hs_run"]
    out = dict(format=np.array(c["hs_chunk_format"]), profiles=np.array(c["profiles"]),
               sides=np.array(run["sides"]), ref=refs, set=np.array(set_label),
               precision=np.array(run["precision"]), batch=np.array(run["batch"]),
               channels_last=np.array(run["channels_last"]), torch=np.array(str(env["torch"])),
               ckpt_bytes=np.array(ck["bytes"]), ckpt_sha256=np.array(ck["sha256"]),
               katago_commit=np.array(c["katago_commit"]), hs_lib_sha256=np.array(c["hs_lib_sha256"]),
               created_utc=np.array(created_utc), seconds=np.array(seconds), cudnn=np.array(str(env["cudnn"])),
               gpu_name=np.array(str(env["gpu"])), driver=np.array(str(env.get("driver"))),
               keys_sha256=np.array(keys_sha), questions_sha256=np.array(questions_sha),
               tool_sha256=np.array(tool_sha), mode=np.array(mode))
    out.update(res)
    return out


def p1_settings(M, fake):
    """tools/t08_rank.py run_settings + the torch field of a fake run."""
    run = M["constants"]["p1_run"]
    if fake:
        return {"precision": "fake-fp32", "batch": int(run["batch"]), "channels_last": bool(run["channels_last"]),
                "device": "cpu"}
    return {"precision": str(run["precision"]), "batch": int(run["batch"]),
            "channels_last": bool(run["channels_last"]), "device": str(run["device"])}


def p1_fields(*, M, refs, set_label, env, ck, torch_v, settings, meta_sha, code_json, keys_sha, questions_sha,
              tool_sha, mode, created_utc, seconds, gpu_seconds, res):
    """tools/t08_rank.py p1_run's save_npz keyword order, field for field."""
    c = M["constants"]
    run = c["p1_run"]
    out = dict(format=np.array(c["p1_chunk_format"]), set=np.array(set_label), sides=np.array(run["sides"]),
               ref=refs, precision=np.array(settings["precision"]), batch=np.array(run["batch"]),
               channels_last=np.array(run["channels_last"]), device=np.array(settings["device"]),
               torch=np.array(str(torch_v)), cudnn=np.array(str(env["cudnn"])), gpu_name=np.array(str(env["gpu"])),
               driver=np.array(str(env.get("driver"))), ckpt_bytes=np.array(ck["bytes"]),
               ckpt_sha256=np.array(ck["sha256"]), katago_commit=np.array(c["katago_commit"]),
               meta_sha256=np.array(meta_sha), code_sha256=np.array(code_json), keys_sha256=np.array(keys_sha),
               questions_sha256=np.array(questions_sha), tool_sha256=np.array(tool_sha), mode=np.array(mode),
               created_utc=np.array(created_utc), seconds=np.array(seconds), gpu_seconds=np.array(gpu_seconds))
    out.update(res)
    return out


def save_chunk(path, fields, tmp_dir=None):
    """np.savez to a local temporary file, copied next to the target and renamed (atomic on Drive and locally)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    base = os.path.basename(path)
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        local = os.path.join(tmp_dir, base + ".tmp.npz")
        np.savez(local, **fields)
        shutil.copyfile(local, path + ".tmp.npz")
        os.remove(local)
    else:
        np.savez(path + ".tmp.npz", **fields)
    os.replace(path + ".tmp.npz", path)
    return path


def check_existing(path, stage, M, refs, set_rec, mode, env, fake, ckpt_sha, code_json=None, meta_sha=None):
    """A chunk on disk is reused only when it is this run's (else Refused) and of this environment (else Drift)."""
    c = M["constants"]
    try:
        with np.load(path, allow_pickle=False) as z:
            got = {k: str(z[k]) for k in ("format", "keys_sha256", "questions_sha256", "mode", "torch", "cudnn",
                                          "gpu_name", "ckpt_sha256", "set")}
            got["sides"] = [str(x) for x in z["sides"]]
            same_refs = np.array_equal(np.asarray(z["ref"]), refs)
            if stage == "hs":
                got["profiles"] = [str(x) for x in z["profiles"]]
            else:
                got.update(code=str(z["code_sha256"]), meta=str(z["meta_sha256"]), precision=str(z["precision"]),
                           device=str(z["device"]), batch=int(z["batch"]), channels_last=bool(z["channels_last"]))
    except Exception as e:  # noqa: BLE001
        raise Refused(f"{path}: unreadable chunk ({e}); move it away and rerun")
    run = c["hs_run"] if stage == "hs" else c["p1_run"]
    want = {"format": c["hs_chunk_format"] if stage == "hs" else c["p1_chunk_format"],
            "keys_sha256": set_rec["keys_sha256"], "questions_sha256": set_rec["questions_sha256"], "mode": mode,
            "set": CHUNK_SET_PREFIX[stage] + set_rec["set"], "sides": list(run["sides"]), "ckpt_sha256": ckpt_sha}
    if stage == "hs":
        want["profiles"] = list(c["profiles"])
    else:
        want.update(code=code_json, meta=meta_sha, **p1_settings(M, fake))
    diff = [k for k in want if got[k] != want[k]] + ([] if same_refs else ["ref"])
    if diff:
        raise Refused(f"{path}: an existing chunk differs from this run ({diff}); never overwritten")
    envd = {k: (got[k], str(env[e])) for k, e in (("torch", "torch"), ("cudnn", "cudnn"), ("gpu_name", "gpu"))
            if got[k] != str(env[e])}
    if envd:
        raise Drift(f"{path}: written with another environment {envd} (resume only on the same GPU / torch / cuDNN)")


# ================================================================ the network side (kit modules)
def kit_import(kit, stage, M, fake, tools=None):
    """Put the kit's pinned code on sys.path (after the SHA-256 checks of verify_kit) and import it. -> modules."""
    pilot = os.path.join(kit, *PILOT_DIR)
    p1 = os.path.join(kit, *P1_DIR)
    for rel, want in M["pins_lf"].items():
        if sha_lf(kit_path(kit, rel)) != want:
            raise Refused(f"{rel} is not the pinned file")
    for d in ((p1, pilot) if stage == "p1" else (pilot,)):
        if d not in sys.path:
            sys.path.insert(0, d)
    mods = {}
    if stage == "hs":
        import pilot_run as PR
        if os.path.normcase(os.path.dirname(os.path.abspath(PR.__file__))) != os.path.normcase(os.path.abspath(pilot)):
            raise Refused(f"pilot_run imported from {PR.__file__}")
        mods.update(PR=PR, H=PR.H)
    else:
        import p1_lib as L1
        import p1_extract as E
        for mod in (L1, E):
            if os.path.normcase(os.path.dirname(os.path.abspath(mod.__file__))) != os.path.normcase(os.path.abspath(p1)):
                raise Refused(f"{mod.__name__} imported from {mod.__file__}")
        mods.update(L1=L1, E=E, H=L1.H)
    if os.path.normcase(os.path.dirname(os.path.abspath(mods["H"].__file__))) != os.path.normcase(os.path.abspath(pilot)):
        raise Refused(f"hs_lib imported from {mods['H'].__file__}")
    if fake:
        if not tools or not os.path.exists(os.path.join(tools, "p07_hs.py")):
            raise Refused("--fake-scorer needs --tools (the repository's tools/; local smoke only, never on Colab)")
        tools = os.path.abspath(tools)
        if tools not in sys.path:
            sys.path.append(tools)
    return mods


def make_scorer(stage, M, mods, fake, log):
    """-> (scorer, checkpoint record, torch version, group_positions). Real: CUDA, the full checkpoint SHA-256, the
    frozen settings (t07_rank hs_run / t08_rank p1_run)."""
    c = M["constants"]
    if stage == "hs":
        run = c["hs_run"]
        gp = max(1, run["batch"] // len(c["profiles"])) * run["group_batches"]
        if fake:
            import p07_hs as h7
            return h7.FakeScorer(len(c["profiles"])), {"bytes": c["ckpt_bytes"], "sha256": "fake"}, "fake", gp
    else:
        run = c["p1_run"]
        gp = run["batch"] * run["group_batches"]
        if fake:
            import p08_p1 as e8
            return e8.FakeProbeScorer(), {"bytes": c["ckpt_bytes"], "sha256": "fake"}, "fake", gp
    import torch
    if not torch.cuda.is_available():
        raise Refused("CUDA is not available (pick a GPU runtime)")
    H = mods["H"]
    ck = H.check_checkpoint(H.CKPT_PATH, full_hash=True)
    if ck.get("sha256") != c["ckpt_sha256"]:
        raise Refused(f"checkpoint SHA-256 {ck.get('sha256')} is not {c['ckpt_sha256']}")
    torch.backends.cudnn.benchmark = run["cudnn_benchmark"]
    if stage == "hs":
        model, _st = H.load_model(H.CKPT_PATH, device="cuda", check=False)
        sc = H.Scorer(model, H.profiles_from_names(c["profiles"]), device="cuda", precision=run["precision"],
                      batch=run["batch"], channels_last=run["channels_last"])
    else:
        L1 = mods["L1"]
        model = L1.load_model(device=run["device"])
        sc = L1.ProbeScorer(model, device=run["device"], precision=run["precision"], batch=run["batch"],
                            channels_last=run["channels_last"])
    log(f"{stage}: model loaded on {torch.cuda.get_device_name(0)} (checkpoint SHA-256 checked)")
    return sc, ck, torch.__version__, gp


# ================================================================ one set
def run_set(kit, M, I, out, stage, s, *, workers, fake, max_games=0, stop_after_min=0, tmp_dir=None, log=print,
            mods=None, scorer_cache=None):
    """Compute (or resume) the chunks of one set of one stage. -> {'done': bool, 'chunks': n, ...}."""
    mode = I["mode"]
    if fake and mode in ("real", "rehearsal"):
        raise Refused("--fake-scorer is never used for real (test-day) or rehearsal inputs")
    set_rec = dict(I["sets"][s], set=s)
    games = I["sets_data"][s]["games"]
    if max_games:
        if mode == "real":
            raise Refused("--max-games is for drills only")
        games = games[:max_games]
    c = M["constants"]
    size = c["hs_run"]["chunk"] if stage == "hs" else c["p1_run"]["chunk"]
    env = environment(fake)
    why = allowed_problem(I, stage, env)
    if why:
        raise Refused(why)
    chunk_dir = os.path.join(out, stage, s)
    os.makedirs(chunk_dir, exist_ok=True)
    run_log = os.path.join(chunk_dir, "RUN_LOG.jsonl")
    st_env = stage_environment(out, stage, env, I["input_id"], mode)
    refs_all = np.arange(len(games), dtype=np.int64)
    plan = chunk_plan(len(games), size)
    mods = mods or kit_import(kit, stage, M, fake)
    code_json = meta_sha = None
    if stage == "p1":
        L1 = mods["L1"]
        code = L1.code_shas()
        if code != c["p1_code_record"]:
            raise Refused(f"the P1 code record of the kit {code} is not the registered one {c['p1_code_record']}")
        code_json = json.dumps(code, sort_keys=True)
        meta_sha = L1.neutral_table_sha()
        if meta_sha != c["neutral_meta_sha256"]:
            raise Refused(f"neutral metadata SHA-256 {meta_sha} is not {c['neutral_meta_sha256']}")
    ckpt_sha = "fake" if fake else c["ckpt_sha256"]
    todo = []
    for ci, b, e in plan:
        p = os.path.join(chunk_dir, chunk_name(ci))
        if os.path.exists(p):
            check_existing(p, stage, M, refs_all[b:e], set_rec, mode, env, fake, ckpt_sha, code_json, meta_sha)
        else:
            todo.append((ci, b, e))
    want = {chunk_name(ci) for ci, _, _ in plan}
    extra = sorted(f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".npz")
                   and ".tmp" not in f and f not in want)
    if extra:
        raise Refused(f"{chunk_dir} holds chunks that are not this run's: {extra[:5]}")
    log_line(run_log, msg=f"{stage} start", set=s, mode=mode, games=len(games), chunks=len(plan), todo=len(todo),
             run_id=st_env["run_id"], environment=env_key(env))
    tool_sha = sha_lf(os.path.abspath(__file__))
    label = CHUNK_SET_PREFIX[stage] + s
    stopped = None
    if todo:
        key = (stage, fake)
        if scorer_cache is not None and key in scorer_cache:
            sc, ck, torch_v, gp = scorer_cache[key]
        else:
            sc, ck, torch_v, gp = make_scorer(stage, M, mods, fake, log)
            if scorer_cache is not None:
                scorer_cache[key] = (sc, ck, torch_v, gp)
        import multiprocessing as mp
        t_run = time.time()
        with mp.get_context("spawn").Pool(workers) as pool:
            for k, (ci, b, e) in enumerate(todo):
                if os.path.exists(os.path.join(chunk_dir, "STOP")):
                    stopped = "STOP file"
                elif stop_after_min and time.time() - t_run > stop_after_min * 60:
                    stopped = f"--stop-after-min {stop_after_min}"
                if stopped:
                    log_line(run_log, msg="stopping before the next chunk (run the same command to continue)",
                             reason=stopped, chunks_left=len(todo) - k)
                    break
                t1 = time.time()
                path = os.path.join(chunk_dir, chunk_name(ci))
                if stage == "hs":
                    res = mods["PR"].run_chunk(pool, sc, games[b:e], tuple(c["hs_run"]["sides"]), gp)
                    secs = time.time() - t1
                    fields = hs_fields(M=M, refs=refs_all[b:e], set_label=label, env=env, ck=ck,
                                       keys_sha=set_rec["keys_sha256"], questions_sha=set_rec["questions_sha256"],
                                       tool_sha=tool_sha, mode=mode, created_utc=now_iso(), seconds=secs, res=res)
                    work = {"evals": int(len(res["pos_side"]) * len(c["profiles"]))}
                else:
                    res, t_gpu = mods["E"].run_chunk(pool, sc, games[b:e], gp)
                    secs = time.time() - t1
                    fields = p1_fields(M=M, refs=refs_all[b:e], set_label=label, env=env, ck=ck, torch_v=torch_v,
                                       settings=p1_settings(M, fake), meta_sha=meta_sha, code_json=code_json,
                                       keys_sha=set_rec["keys_sha256"], questions_sha=set_rec["questions_sha256"],
                                       tool_sha=tool_sha, mode=mode, created_utc=now_iso(), seconds=secs,
                                       gpu_seconds=t_gpu, res=res)
                    work = {"positions": int(len(res["pos_side"]))}
                save_chunk(path, fields, tmp_dir)
                el = time.time() - t_run
                log_line(run_log, msg="chunk done", stage=stage, set=s, chunk=ci, games=e - b,
                         failed=int(res["failed"].sum()), seconds=round(secs, 1), sha256=sha_raw(path),
                         progress=f"{k + 1}/{len(todo)}",
                         eta_min=round(el / (k + 1) * (len(todo) - k - 1) / 60, 1), **work)
    done = not stopped
    files = [chunk_name(ci) for ci, _, _ in plan]
    if done:
        missing = [f for f in files if not os.path.exists(os.path.join(chunk_dir, f))]
        if missing:
            raise Refused(f"{stage} {s}: chunks missing after the run: {missing[:5]}")
    return {"stage": stage, "set": s, "done": done, "stopped": stopped, "games": len(games), "chunks": len(plan),
            "computed_now": len(todo) if done else None, "run_id": st_env["run_id"], "environment": env_key(env),
            "chunk_sha256": {f: sha_raw(os.path.join(chunk_dir, f)) for f in files
                             if os.path.exists(os.path.join(chunk_dir, f))}}


# ================================================================ commands
def cmd_verify(a):
    M = load_kit_manifest(a.kit)
    rec = verify_kit(a.kit, M, ckpt_hash=not a.no_ckpt_hash, log=print)
    if a.input:
        I = load_input(a.input, M)
        rec["input"] = {"input_id": I["input_id"], "mode": I["mode"], "runs": I["runs"],
                        "sets": {k: v["games"] for k, v in I["sets"].items()}}
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return 0


def cmd_run(a):
    t0 = time.time()
    M = load_kit_manifest(a.kit)
    I = load_input(a.input, M)
    kit_rec = verify_kit(a.kit, M, ckpt_hash=not a.fake_scorer, log=print)
    runs = [r for r in I["runs"] if (not a.stage or r["stage"] == a.stage) and (not a.set or r["set"] == a.set)]
    if not runs:
        raise Refused(f"no run of this input matches --stage {a.stage} --set {a.set}")
    workers = a.workers or max(1, min(8, os.cpu_count() or 1))
    if a.fake_scorer and I["mode"] in ("real", "rehearsal"):
        raise Refused("--fake-scorer is never used for real (test-day) or rehearsal inputs")
    env0 = environment(a.fake_scorer)            # before anything is written (review major 1)
    for st in dict.fromkeys(r["stage"] for r in runs):
        why = allowed_problem(I, st, env0)
        if why:
            raise Refused(why)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    for f in ("INPUT_MANIFEST.json",):
        shutil.copyfile(os.path.join(a.input, f), os.path.join(out, f))
    shutil.copyfile(os.path.join(a.kit, "KIT_MANIFEST.json"), os.path.join(out, "KIT_MANIFEST.json"))
    results, code = [], 0
    mods_cache, scorer_cache = {}, {}
    for r in runs:
        stage = r["stage"]
        if stage not in mods_cache:
            mods_cache[stage] = kit_import(a.kit, stage, M, a.fake_scorer, a.tools)
        res = run_set(a.kit, M, I, out, stage, r["set"], workers=workers, fake=a.fake_scorer,
                      max_games=a.max_games, stop_after_min=a.stop_after_min, tmp_dir=a.tmp, mods=mods_cache[stage],
                      scorer_cache=scorer_cache)
        results.append(res)
        if not res["done"]:
            code = EXIT_STOPPED
            break
    for sc, *_ in scorer_cache.values():
        if hasattr(sc, "close"):
            sc.close()
    summary = {"format": SUMMARY_FORMAT, "core": CORE_FORMAT, "input_id": I["input_id"], "mode": I["mode"],
               "kit_id": M.get("kit_id"), "kit": kit_rec, "finished_utc": now_iso(), "exit_code": code,
               "seconds": round(time.time() - t0, 1), "runs": results, "core_sha256_lf": sha_lf(os.path.abspath(__file__)),
               "fake_scorer": bool(a.fake_scorer), "max_games": a.max_games or None}
    write_json_atomic(os.path.join(out, "RUN_SUMMARY.json"), summary)
    print(f"RUN_SUMMARY: exit {code}; " + "; ".join(f"{x['stage']} {x['set']}: {'done' if x['done'] else 'stopped'} "
                                                     f"({x['chunks']} chunks)" for x in results), flush=True)
    return code


def build_parser():
    ap = argparse.ArgumentParser(prog="gpu_fallback_core.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--kit", required=True)
    v.add_argument("--input")
    v.add_argument("--no-ckpt-hash", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--kit", required=True)
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--stage", choices=STAGES)
    r.add_argument("--set")
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--stop-after-min", type=float, default=0)
    r.add_argument("--fake-scorer", action="store_true")
    r.add_argument("--max-games", type=int, default=0)
    r.add_argument("--tmp")
    r.add_argument("--tools", help="the repository's tools/ folder (only with --fake-scorer: p07_hs.FakeScorer, "
                                   "p08_p1.FakeProbeScorer)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return cmd_verify(a) if a.cmd == "verify" else cmd_run(a)
    except (Refused, Drift) as e:
        print(f"{'REFUSED' if e.code == EXIT_REFUSED else 'ENVIRONMENT DIFFERS'} (exit {e.code}): {e}", file=sys.stderr,
              flush=True)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
