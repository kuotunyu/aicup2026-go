"""PREREG_07 competition 2: KataGo HumanSL per-game scores for the formal steps (docs/PREREG_07_HUMANSL.md rev 1,
sections 1.2-1.7 and 2.2; results/prereg07/design/spec.md sections 0, 2, 3).

    CPU  PY=.venv/Scripts/python.exe      GPU  GPY=.venv-gpu/Scripts/python.exe     (Git Bash, PYTHONDONTWRITEBYTECODE=1)
    PY  tools/p07_hs.py manifest [--precommit]      -> results/prereg07/hs/inner_manifest.json
    GPY tools/p07_hs.py val                          -> data/prereg07/val/chunk_*.npz, then results/prereg07/hs/
                                                        val_hs.npz, val_seam_moves.npz, val_report.json
    PY  tools/p07_hs.py status                       (read only: progress, STOP file, estimate)
    smoke (never the validation file): GPY tools/p07_hs.py val --smoke --questions Q.csv --out-dir data/prereg07/tmp/<x>
                                       [--max-games N] [--fake-scorer] [--workers W] [--ignore-other-gpu-jobs]

manifest  The pilot chunks of data/prereg07/pilot/{fitmap,bank,mono} (15 / 60 / 307): SHA-256, games, failed, no target
          move, refs equal to the plan (pilot_run.set_games), metadata equal to PREREG_07 1.2-1.4, the full checkpoint
          SHA-256, and the move cross-check of every bank / mono game against results/prereg04/opp/inner_o4.npz (another
          replay program on the same inner_keys rows; both sides not failed: HS n_moves = O4 n_moves total and HS
          target moves = O4 n_moves[:, 0, :] total). Reads chunk metadata, counts and SHA-256 only; computes no score.
val       Only after results/prereg07/rank/frozen_config.json (gates passed, score_pending_val_hs) and hs_map.npz are
          committed and clean, the p07 tools and the four pinned files committed, clean and equal to what tune recorded,
          no val_report.json yet, no other compute process on the GPU, and .venv-gpu's python / torch / cuDNN / GPU
          equal to frozen_config.json's reference. Then, in this order: read data/local/rank_val_questions.csv (its
          question_id / num_games / color_i / sgf_i columns, p04_opp.read_question_slots) and val_keys.npz
          (p02 load_val_keys), check them row by row (p04_opp.slots_match_keys: 1,589 questions, 19,156 games); only
          then import hs_lib / pilot_run (their guard closes data/local for the rest of the process). Games in
          val_keys order; target colour = the question file's colour; pilot_run.run_chunk per 200 games, written
          atomically to data/prereg07/val/chunk_<i>.npz. Rerunning the same command resumes (chunks that pass the
          check are skipped; a chunk with another environment or key file is refused, never overwritten). STOP file
          in data/prereg07/val/ or --stop-after-min end the run after the chunk in progress. When every chunk is
          there: val_hs.npz, val_seam_moves.npz (the first 400 games' per-move log p) and val_report.json. Failed
          games > 0 -> status stop, exit 1 (PREREG_07 2.7). Start it as a detached background process (spec 0): the
          script runs this one command and nothing after it.
Smoke     a question file outside data/local built from training games (or synthetic SGFs); output under
          data/prereg07/tmp/; --fake-scorer replaces the network with a deterministic CPU function (tests).
Exit codes: 0 done (or stopped early on request); 1 status stop; 2 refused or error.
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FORMAT = "c10-prereg07 pilot chunk v1"         # PREREG_07 1.2 / spec 3.3: the pilot format, plus fields
VAL_DIR = os.path.join(REPO, "data", "prereg07", "val")
GPU_NAME = "NVIDIA GeForce RTX 4090"


def P():
    """tools/p07_rank_fuse.py (constants and helpers; imported lazily to keep the spawn children light)."""
    import p07_rank_fuse as pf
    return pf


class HSRefused(RuntimeError):
    pass


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def gpu_compute_processes():
    """nvidia-smi process rows of Type exactly 'C' (same parsing as hs_lib.other_compute_processes; own copy so that
    hs_lib is not imported before the question file is read)."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)
    rows, in_proc = [], False
    for line in out.splitlines():
        if "Processes:" in line:
            in_proc = True
            continue
        if in_proc:
            m = re.match(r"\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(C|G|C\+G)\s+(.*?)\s+(\S+)\s*\|", line)
            if m and m.group(2) == "C":
                rows.append({"pid": int(m.group(1)), "name": m.group(3).strip()})
    return rows, out


def gpu_env(fake=False):
    if fake:
        return {"python": sys.version.split()[0], "torch": "fake", "cudnn": "fake", "gpu": "fake", "driver": "fake",
                "numpy": np.__version__}
    import torch
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
           "cuda": torch.cuda.is_available(), "cudnn": torch.backends.cudnn.version(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "driver": P().nvidia_query().get("driver")}
    return env


# ================================================================ manifest (spec 3.1)
def cmd_manifest(a):
    t0 = time.time()
    pf = P()
    out = pf.MANIFEST
    if os.path.exists(out):
        if not (a.rerun and a.reason):
            raise HSRefused(f"{pf.rel(out)} exists; rerun only with --rerun --reason TEXT")
        if pf.git(["ls-files", "--error-unmatch", "--", pf.rel(out)])[0] == 0:
            raise HSRefused("inner_manifest.json is committed; it is never overwritten")
    ts = pf.tool_state()
    pf.require_tools(ts, a.precommit)
    pf.check_pins()
    if pf.PILOT_CODE not in sys.path:
        sys.path.insert(0, pf.PILOT_CODE)
    import pilot_run as PR    # imports hs_lib (guard); manifest never needs data/local
    H = PR.H
    plan_sha = pf.sha_raw(pf.PLAN)
    if plan_sha != pf.PLAN_SHA:
        raise HSRefused(f"plan.json SHA-256 {plan_sha} != {pf.PLAN_SHA}")
    plan = PR.load_plan()
    if plan["profiles"] != pf.PROFILES or plan["chunk"] != pf.RUN["chunk"]:
        raise HSRefused("plan.json profiles / chunk differ from PREREG_07 1.4")
    print("checkpoint: full SHA-256 ...", flush=True)
    ck = H.check_checkpoint(H.CKPT_PATH, full_hash=True)
    sets, refs_ok, torch_vals = {}, {}, set()
    inner_rows = {}
    for s in ("fitmap", "bank", "mono"):
        games = PR.set_games(plan, s)
        refs_all = np.array([g["ref"] for g in games], dtype=np.int64)
        nch = (len(games) + plan["chunk"] - 1) // plan["chunk"]
        files = sorted(f for f in glob.glob(os.path.join(pf.PILOT_DATA, s, "chunk_*.npz")) if ".tmp" not in f)
        rows, ok_all = [], True
        tot = {"games": 0, "failed": 0, "no_target_moves": 0, "no_move": 0, "failed_or_no_move": 0}
        for ci in range(nch):
            p = os.path.join(pf.PILOT_DATA, s, f"chunk_{ci:05d}.npz")
            st, en = ci * plan["chunk"], min(len(games), (ci + 1) * plan["chunk"])
            if not os.path.exists(p):
                ok_all = False
                continue
            with np.load(p, allow_pickle=False) as z:
                ref = np.asarray(z["ref"])
                failed = np.asarray(z["failed"], dtype=bool)
                cnt = np.asarray(z["counts"])[:, 0]
                nm = np.asarray(z["n_moves"])
                meta = {"profiles": [str(x) for x in z["profiles"]], "sides": [str(x) for x in z["sides"]],
                        "precision": str(z["precision"]), "batch": int(z["batch"]),
                        "channels_last": bool(z["channels_last"]), "katago_commit": str(z["katago_commit"]),
                        "hs_lib_sha256": str(z["hs_lib_sha256"]), "format": str(z["format"])}
                rec = {"file": pf.rel(p), "sha256": pf.sha_raw(p), "games": int(len(ref)), "failed": int(failed.sum()),
                       "no_target_moves": int(((cnt == 0) & ~failed).sum()), "no_move": int((nm == 0).sum()),
                       "created_utc": str(z["created_utc"]), "torch": str(z["torch"]),
                       "ckpt_sha256_in_chunk": str(z["ckpt_sha256"])}
            torch_vals.add(rec["torch"])
            rec["ref_matches_plan"] = bool(np.array_equal(ref, refs_all[st:en]))
            rec["meta_matches"] = meta == {"profiles": pf.PROFILES, "sides": pf.RUN["sides"],
                                           "precision": pf.RUN["precision"], "batch": pf.RUN["batch"],
                                           "channels_last": pf.RUN["channels_last"],
                                           "katago_commit": pf.KATAGO_COMMIT,
                                           "hs_lib_sha256": pf.PINNED["results/prereg07/explore/pilot_humansl/"
                                                                      "hs_lib.py"], "format": FORMAT}
            ok_all = ok_all and rec["ref_matches_plan"] and rec["meta_matches"]
            for k in ("games", "failed", "no_target_moves", "no_move"):
                tot[k] += rec[k]
            tot["failed_or_no_move"] += int((failed | (nm == 0)).sum())
            rows.append(rec)
            if s in ("bank", "mono"):
                inner_rows[s] = inner_rows.get(s, []) + [(ref, failed, cnt, nm)]
        sets[s] = {"chunks": rows, "chunks_expected": nch, "chunks_present": len(rows), "games_planned": len(games),
                   "totals": tot, "extra_files": sorted(pf.rel(f) for f in files
                                                         if os.path.basename(f) not in
                                                         {os.path.basename(r["file"]) for r in rows})}
        refs_ok[s] = ok_all and len(rows) == nch
        print(f"{s}: {len(rows)}/{nch} chunks, {tot}", flush=True)
    # move cross-check (PREREG_07 2.2 item 5) against inner_o4.npz on the same inner_keys rows
    with np.load(pf.INNER_KEYS, allow_pickle=False) as z:
        keys = {k: np.asarray(z[k]).astype(str) for k in ("rank", "game_id", "player_id", "color")}
    with np.load(pf.INNER_O4, allow_pickle=False) as z:
        o4 = {k: np.asarray(z[k]) for k in ("rank", "game_id", "player_id", "color", "n_moves", "anomalies")}
    key_diff = [k for k in ("rank", "game_id", "player_id", "color") if not np.array_equal(o4[k].astype(str), keys[k])]
    xc_parts = {}
    for s, parts in inner_rows.items():
        ref = np.concatenate([p[0] for p in parts])
        failed = np.concatenate([p[1] for p in parts])
        cnt = np.concatenate([p[2] for p in parts])
        nm = np.concatenate([p[3] for p in parts])
        nmo = o4["n_moves"][ref]
        xc_parts[s] = pf.move_cross_check(nm, cnt, failed, nmo.sum(axis=(1, 2)), nmo[:, 0, :].sum(1),
                                          o4["anomalies"][ref] > 0,
                                          [f"{keys['rank'][i]}/{keys['game_id'][i]}" for i in ref])
    xc = {"against": pf.rel(pf.INNER_O4), "inner_o4_sha256": pf.sha_raw(pf.INNER_O4),
          "key_arrays_equal_inner_keys": not key_diff, "key_arrays_differing": key_diff, "by_set": xc_parts,
          "compared": sum(v["compared"] for v in xc_parts.values()),
          "mismatches": (sum(v["mismatches"] for v in xc_parts.values()) if not key_diff else None),
          "definition": "both sides not failed (HS failed false, O4 anomalies 0): HS n_moves == sum O4 n_moves, HS "
                        "target moves (counts[:, 0]) == sum O4 n_moves[:, 0, :]"}
    complete = all(sets[s]["chunks_present"] == pf.SET_CHUNKS[s] for s in sets)
    man = {"format": pf.FORMATS["manifest"], "what": "PREREG_07 2.2 / spec 3.1: pilot chunk manifest (no score)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), "precommit": bool(a.precommit),
           "complete": bool(complete), "plan_sha256": plan_sha, "checkpoint": ck,
           "refs_match_plan": {**refs_ok, "all": all(refs_ok.values())}, "torch_values": sorted(torch_vals),
           "sets": sets, "move_cross_check": xc, "tool": ts, "pinned_sha256": pf.check_pins(),
           "environment": pf.environment_cpu()}
    pf.write_json(out, man)
    print(f"manifest: complete {complete}; cross-check {xc['compared']} games, {xc['mismatches']} mismatches -> "
          f"{pf.rel(out)}")
    return 0


# ================================================================ val (spec 3.2, 3.3)
class FakeScorer:
    """Deterministic CPU stand-in for hs_lib.Scorer (smoke tests only): log p of the played move is a fixed function
    of the move index, the legal count and the profile index."""

    def __init__(self, n_profiles):
        self.N = n_profiles

    def logp_moves(self, feats):
        T = len(feats["move"])
        out = np.zeros((T, self.N), dtype=np.float32)
        if T:
            nleg = feats["legal"].sum(1).astype(np.float64)
            mv = feats["move"].astype(np.float64)
            for j in range(self.N):
                out[:, j] = (-np.log(nleg) - 0.01 * (j + 1) * ((mv % 7) / 7.0)).astype(np.float32)
        return out


def chunk_state(path, refs, keys_sha, env):
    """-> 'ok' (reuse), or raises when a chunk exists but cannot be reused (never overwritten)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            got = {"format": str(z["format"]), "profiles": [str(x) for x in z["profiles"]],
                   "sides": [str(x) for x in z["sides"]], "keys_sha256": str(z["keys_sha256"]),
                   "torch": str(z["torch"]), "cudnn": str(z["cudnn"]), "gpu_name": str(z["gpu_name"]),
                   "ckpt_sha256": str(z["ckpt_sha256"])}
            same_refs = np.array_equal(np.asarray(z["ref"]), refs)
    except Exception as e:  # noqa: BLE001
        raise HSRefused(f"{path}: unreadable chunk ({e}); move it away and rerun")
    pf = P()
    want = {"format": FORMAT, "profiles": pf.PROFILES, "sides": pf.RUN["sides"], "keys_sha256": keys_sha}
    diff = [k for k in want if got[k] != want[k]] + ([] if same_refs else ["ref"])
    if diff:
        raise HSRefused(f"{path}: an existing chunk differs from this run ({diff}); never overwritten")
    envd = [k for k, e in (("torch", "torch"), ("cudnn", "cudnn"), ("gpu_name", "gpu")) if got[k] != str(env[e])]
    if envd:
        raise HSRefused(f"{path}: written with another environment ({envd}: {[got[k] for k in envd]}); stop and "
                        f"find out why (spec 3.3)")
    if got["ckpt_sha256"] not in (P().CKPT_SHA, "fake"):
        raise HSRefused(f"{path}: chunk without the full checkpoint SHA-256")
    return "ok"


def val_prerequisites(a, pf):
    """Real mode, before data/local is opened -> (frozen config, records)."""
    fpath = os.path.join(pf.OUT, "frozen_config.json")
    mpath = os.path.join(pf.OUT, "hs_map.npz")
    rec = {"frozen": pf.require_committed([fpath, mpath], "frozen_config.json / hs_map.npz (commit them first)")}
    fc = pf.read_json(fpath)
    if fc.get("format") != pf.FORMATS["frozen"] or fc.get("preview") or fc.get("score_pending_val_hs") is not True:
        raise HSRefused("frozen_config.json is not a passed real frozen configuration (score_pending_val_hs)")
    if pf.sha_raw(mpath) != fc["hs"]["hs_map_npz_sha256"]:
        raise HSRefused("hs_map.npz differs from the SHA-256 frozen_config.json records")
    rec["design_records"] = pf.design_records_bound(fc)
    ts = pf.tool_state()
    pf.require_tools(ts, False)
    diff = pf.tools_equal_record(ts, fc.get("tool"))
    if diff:
        raise HSRefused(f"tool files differ from the ones tune recorded: {diff}")
    rep = os.path.join(pf.HS_OUT, "val_report.json")
    if os.path.exists(rep):
        raise HSRefused("val_report.json exists: the registered validation HS runs once (resume, never rerun)")
    procs, raw = gpu_compute_processes()
    if procs is None or procs:
        raise HSRefused(f"nvidia-smi lists another compute process (or failed): {procs}")
    env = gpu_env()
    ref = fc["environment"]["venv_gpu_reference"]
    bad = [k for k in ("python", "torch", "cudnn", "gpu") if str(env[k]) != str(ref[k])]
    if bad:
        raise HSRefused(f".venv-gpu differs from frozen_config.json ({bad}: {[env[k] for k in bad]} vs "
                        f"{[ref[k] for k in bad]})")
    rec.update(mode="real", tool=ts, environment=env, frozen_config_sha256=pf.sha_lf(fpath),
               frozen_commit_time=pf.commit_time(fpath), nvidia_compute_processes_before=procs)
    return fc, rec


def cmd_val(a):
    t0 = time.time()
    pf = P()
    if a.smoke:
        if not a.questions or not a.out_dir:
            raise HSRefused("--smoke needs --questions (outside data/local) and --out-dir (under data/prereg07/tmp)")
        qfile = os.path.abspath(a.questions)
        if pf.under_data_local(qfile):
            raise HSRefused("smoke runs never read data/local; a question file there needs the real mode")
        out_dir = os.path.abspath(a.out_dir)
        if not pf.inside(out_dir, pf.TMP):
            raise HSRefused(f"--smoke --out-dir must be under {pf.rel(pf.TMP)}")
        chunk_dir, hs_dir = os.path.join(out_dir, "val"), os.path.join(out_dir, "hs")
        if os.path.exists(os.path.join(hs_dir, "val_report.json")):
            raise HSRefused("smoke val_report.json exists; use a fresh --out-dir")
        pre = {"mode": "smoke"}
        fc = None
        env = gpu_env(fake=a.fake_scorer)
        if not a.fake_scorer and not a.ignore_other_gpu_jobs:
            procs, _ = gpu_compute_processes()
            if procs:
                raise HSRefused(f"nvidia-smi lists another compute process: {procs}")
    else:
        if a.questions or a.out_dir or a.max_games or a.fake_scorer or a.ignore_other_gpu_jobs:
            raise HSRefused("real mode takes no --questions / --out-dir / --max-games / --fake-scorer / "
                            "--ignore-other-gpu-jobs")
        fc, pre = val_prerequisites(a, pf)
        env = pre["environment"]
        qfile = pf.QUESTIONS
        chunk_dir, hs_dir = VAL_DIR, pf.HS_OUT
    pf.refuse_label_path(qfile)
    # ---- 1 the question file and the keys (before hs_lib / pilot_run)
    import p04_opp
    import p02_rank_fuse as rf
    q_sha = pf.sha_raw(qfile)
    slots = p04_opp.read_question_slots(qfile)
    if a.smoke:
        keys_sha = q_sha
        vk = {"question_id": [s[0] for s in slots], "slot": [s[1] for s in slots], "color": [s[2] for s in slots],
              "n": len(slots)}
    else:
        if q_sha != pf.QUESTIONS_SHA:
            raise HSRefused(f"question file SHA-256 {q_sha} != {pf.QUESTIONS_SHA}")
        vk = rf.load_val_keys(os.path.dirname(pf.VAL_KEYS))
        if vk["sha256"] != pf.VAL_KEYS_SHA:
            raise HSRefused("val_keys.npz is not the registered file")
        keys_sha = vk["sha256"]
    p04_opp.slots_match_keys(slots, vk)
    n_q = len(dict.fromkeys(vk["question_id"]))
    if not a.smoke and (n_q != pf.VAL_QUESTIONS or vk["n"] != pf.VAL_GAMES):
        raise HSRefused(f"{n_q} questions / {vk['n']} games, registered {pf.VAL_QUESTIONS} / {pf.VAL_GAMES}")
    games = [{"sgf_content": sgf, "color": col, "ref": i} for i, (_q, _s, col, sgf) in enumerate(slots)]
    if a.max_games:
        games = games[:a.max_games]
    del slots
    # ---- 2 only now hs_lib / pilot_run (their guard closes data/local for this process)
    pf.check_pins()
    if pf.PILOT_CODE not in sys.path:
        sys.path.insert(0, pf.PILOT_CODE)
    import pilot_run as PR
    H = PR.H
    import multiprocessing as mp
    os.makedirs(chunk_dir, exist_ok=True)
    refs_all = np.array([g["ref"] for g in games], dtype=np.int64)
    C = pf.RUN["chunk"]
    chunks = [(ci, s, min(len(games), s + C)) for ci, s in enumerate(range(0, len(games), C))]
    todo = []
    for ci, s, e in chunks:
        p = os.path.join(chunk_dir, f"chunk_{ci:05d}.npz")
        if os.path.exists(p):
            chunk_state(p, refs_all[s:e], keys_sha, env)
        else:
            todo.append((ci, s, e))
    print(json.dumps({"time_utc": now_iso(), "msg": "val start", "mode": pre["mode"], "games": len(games),
                      "chunks": len(chunks), "todo": len(todo)}), flush=True)
    stopped = None
    tool_sha = pf.sha_lf(os.path.abspath(__file__))
    hs_sha = pf.PINNED["results/prereg07/explore/pilot_humansl/hs_lib.py"]
    if todo:
        if a.fake_scorer:
            sc, ck = FakeScorer(len(pf.PROFILES)), {"bytes": pf.CKPT_BYTES, "sha256": "fake"}
            torch_v = "fake"
        else:
            import torch
            if not torch.cuda.is_available():
                raise HSRefused("CUDA is not available")
            if env["gpu"] != GPU_NAME:
                raise HSRefused(f"GPU {env['gpu']} is not {GPU_NAME}")
            ck = H.check_checkpoint(H.CKPT_PATH, full_hash=True)
            if ck.get("sha256") != pf.CKPT_SHA:
                raise HSRefused("checkpoint SHA-256 differs")
            torch.backends.cudnn.benchmark = pf.RUN["cudnn_benchmark"]
            model, _st = H.load_model(H.CKPT_PATH, device="cuda", check=False)
            sc = H.Scorer(model, H.profiles_from_names(pf.PROFILES), device="cuda", precision=pf.RUN["precision"],
                          batch=pf.RUN["batch"], channels_last=pf.RUN["channels_last"])
            torch_v = torch.__version__
        group_positions = max(1, pf.RUN["batch"] // len(pf.PROFILES)) * pf.RUN["group_batches"]
        t_run = time.time()
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for k, (ci, s, e) in enumerate(todo):
                if os.path.exists(os.path.join(chunk_dir, "STOP")):
                    stopped = "STOP file"
                elif a.stop_after_min and time.time() - t_run > a.stop_after_min * 60:
                    stopped = f"--stop-after-min {a.stop_after_min}"
                if stopped:
                    print(json.dumps({"time_utc": now_iso(), "msg": "stopping before the next chunk (rerun the same "
                                      "command to continue)", "reason": stopped, "chunks_left": len(todo) - k}),
                          flush=True)
                    break
                t1 = time.time()
                res = PR.run_chunk(pool, sc, games[s:e], tuple(pf.RUN["sides"]), group_positions)
                secs = time.time() - t1
                evals = int(len(res["pos_side"]) * len(pf.PROFILES))
                path = os.path.join(chunk_dir, f"chunk_{ci:05d}.npz")
                tmp = path + ".tmp.npz"
                np.savez(tmp, format=np.array(FORMAT), profiles=np.array(pf.PROFILES), sides=np.array(pf.RUN["sides"]),
                         ref=refs_all[s:e], set=np.array("val" if not a.smoke else "val_smoke"),
                         precision=np.array(pf.RUN["precision"]), batch=np.array(pf.RUN["batch"]),
                         channels_last=np.array(pf.RUN["channels_last"]), torch=np.array(torch_v),
                         ckpt_bytes=np.array(ck["bytes"]), ckpt_sha256=np.array(ck["sha256"]),
                         katago_commit=np.array(pf.KATAGO_COMMIT), hs_lib_sha256=np.array(hs_sha),
                         created_utc=np.array(now_iso()), seconds=np.array(secs), cudnn=np.array(str(env["cudnn"])),
                         gpu_name=np.array(str(env["gpu"])), driver=np.array(str(env["driver"])),
                         keys_sha256=np.array(keys_sha), questions_sha256=np.array(q_sha),
                         tool_sha256=np.array(tool_sha), mode=np.array(pre["mode"]), **res)
                os.replace(tmp, path)
                el = time.time() - t_run
                print(json.dumps({"time_utc": now_iso(), "msg": "chunk done", "chunk": ci, "games": e - s,
                                  "failed": int(res["failed"].sum()), "evals": evals, "seconds": round(secs, 1),
                                  "evals_per_s": round(evals / max(secs, 1e-9), 1),
                                  "progress": f"{k + 1}/{len(todo)}",
                                  "eta_min": round(el / (k + 1) * (len(todo) - k - 1) / 60, 1)}), flush=True)
    if stopped:
        return 0
    return assemble(a, pf, chunk_dir, hs_dir, chunks, vk, keys_sha, q_sha, env, pre, fc, t0)


def assemble(a, pf, chunk_dir, hs_dir, chunks, vk, keys_sha, q_sha, env, pre, fc, t0):
    rows, parts = [], []
    for ci, s, e in chunks:
        p = os.path.join(chunk_dir, f"chunk_{ci:05d}.npz")
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in ("ref", "sums", "counts", "n_moves", "failed", "pos_game", "pos_turn", "pos_side",
                                   "pos_logp", "created_utc", "seconds", "errors")}
        parts.append(d)
        failed = np.asarray(d["failed"], dtype=bool)
        cnt = np.asarray(d["counts"])[:, 0]
        rows.append({"file": pf.rel(p), "sha256": pf.sha_raw(p), "games": int(len(d["ref"])),
                     "failed": int(failed.sum()), "no_target_moves": int(((cnt == 0) & ~failed).sum()),
                     "no_move": int((np.asarray(d["n_moves"]) == 0).sum()), "created_utc": str(d["created_utc"]),
                     "seconds": float(d["seconds"]), "evals": int(len(d["pos_side"]) * len(pf.PROFILES)),
                     "errors": json.loads(str(d["errors"]))})
    ref = np.concatenate([d["ref"] for d in parts])
    n = len(ref)
    if not np.array_equal(ref, np.arange(n)):
        raise HSRefused("chunk refs are not the val_keys rows in order")
    sums = np.concatenate([d["sums"][:, 0, :] for d in parts]).astype(np.float64)
    counts = np.concatenate([d["counts"][:, 0] for d in parts]).astype(np.int64)
    n_moves = np.concatenate([d["n_moves"] for d in parts]).astype(np.int64)
    failed = np.concatenate([d["failed"] for d in parts]).astype(bool)
    no_target = (counts == 0) & ~failed
    os.makedirs(hs_dir, exist_ok=True)
    vh = os.path.join(hs_dir, "val_hs.npz")
    np.savez(vh, question_id=np.array(vk["question_id"][:n]), slot=np.array(vk["slot"][:n], dtype=np.int64),
             color=np.array(vk["color"][:n]), sums=sums, counts=counts, n_moves=n_moves, failed=failed,
             no_target_moves=no_target, profiles=np.array(pf.PROFILES))
    seam_parts = parts[:2]
    offs = [0] + [len(seam_parts[0]["ref"])]
    keep = [np.asarray(d["pos_side"]) == 0 for d in seam_parts]
    sm = os.path.join(hs_dir, "val_seam_moves.npz")
    np.savez(sm, pos_game=np.concatenate([np.asarray(d["pos_game"])[k] + o for d, k, o in zip(seam_parts, keep, offs)]
                                         ).astype(np.int32),
             pos_turn=np.concatenate([np.asarray(d["pos_turn"])[k] for d, k in zip(seam_parts, keep)]).astype(np.int16),
             pos_logp=np.concatenate([np.asarray(d["pos_logp"])[k] for d, k in zip(seam_parts, keep)]).astype(
                 np.float32),
             games=np.array(sum(len(d["ref"]) for d in seam_parts)),
             chunk_sha256=np.array([r["sha256"] for r in rows[:2]]), profiles=np.array(pf.PROFILES))
    fails = int((failed | (n_moves == 0)).sum())
    starts = [datetime.datetime.fromisoformat(r["created_utc"]) - datetime.timedelta(seconds=r["seconds"])
              for r in rows]
    evals = sum(r["evals"] for r in rows)
    secs = sum(r["seconds"] for r in rows)
    fpath = os.path.join(pf.OUT, "frozen_config.json")
    rep = {"format": pf.FORMATS["val_report"], "what": "PREREG_07 1.7 validation HS (no score)",
           "mode": pre["mode"], "created": now_iso(), "status": "ok" if fails == 0 else "stop",
           "games": n, "questions": len(dict.fromkeys(vk["question_id"][:n])), "failed": int(failed.sum()),
           "failed_or_no_move": fails, "no_target_moves": int(no_target.sum()), "chunks": rows,
           "first_chunk_started_utc": min(starts).isoformat(timespec="seconds"),
           "last_chunk_created_utc": max(r["created_utc"] for r in rows), "evals": evals, "gpu_seconds": round(secs, 1),
           "evals_per_s": round(evals / max(secs, 1e-9), 1), "environment": env,
           "frozen_config_sha256": (pf.sha_lf(fpath) if fc is not None else None),
           "frozen_config_commit_time": pre.get("frozen_commit_time"), "prerequisites": pre,
           "questions_sha256": q_sha, "keys_sha256": keys_sha, "val_hs_sha256": pf.sha_raw(vh),
           "val_seam_moves_sha256": pf.sha_raw(sm), "run": pf.RUN, "profiles": pf.PROFILES,
           "tool_sha256": pf.sha_lf(os.path.abspath(__file__)), "seconds_this_call": round(time.time() - t0, 1),
           "next": "commit val_hs.npz, val_seam_moves.npz, val_report.json; then p07_rank_fuse.py apply"
           if fails == 0 else "failed games > 0: stop and find out why (PREREG_07 2.7)"}
    pf.write_json(os.path.join(hs_dir, "val_report.json"), rep)
    print(f"val: {n} games, failed {fails}, {evals} evals in {secs:.0f} s -> {pf.rel(hs_dir)}")
    return 0 if fails == 0 else 1


# ================================================================ status (spec 3.4)
def cmd_status(a):
    pf = P()
    d = os.path.abspath(a.out_dir) if a.out_dir else VAL_DIR
    files = sorted(f for f in glob.glob(os.path.join(d, "chunk_*.npz")) if ".tmp" not in f)
    secs, evals = 0.0, 0
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            secs += float(z["seconds"])
            evals += int(len(z["pos_side"]) * len(pf.PROFILES))
    total = (pf.VAL_GAMES + pf.RUN["chunk"] - 1) // pf.RUN["chunk"]
    rate = evals / secs if secs else None
    left_games = max(0, pf.VAL_GAMES - len(files) * pf.RUN["chunk"])
    est = (left_games * 1561 / rate / 3600) if rate else None
    out = {"folder": pf.rel(d) if pf.inside(d, REPO) else d, "chunks_present": len(files), "chunks_total": total,
           "stop_file": os.path.exists(os.path.join(d, "STOP")), "evals_per_s": rate,
           "estimate_hours_left": est, "val_report": os.path.exists(os.path.join(pf.HS_OUT, "val_report.json"))}
    print(json.dumps(out, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--precommit", action="store_true")
    m.add_argument("--rerun", action="store_true")
    m.add_argument("--reason", default=None)
    v = sub.add_parser("val")
    v.add_argument("--smoke", action="store_true")
    v.add_argument("--questions", default=None)
    v.add_argument("--out-dir", default=None)
    v.add_argument("--max-games", type=int, default=0)
    v.add_argument("--fake-scorer", action="store_true")
    v.add_argument("--workers", type=int, default=8)
    v.add_argument("--stop-after-min", type=float, default=0)
    v.add_argument("--ignore-other-gpu-jobs", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "val" and not a.smoke and (a.max_games or a.fake_scorer):
        print("ERROR: --max-games / --fake-scorer are smoke-only", file=sys.stderr)
        return 2
    try:
        return {"manifest": cmd_manifest, "val": cmd_val, "status": cmd_status}[a.cmd](a)
    except HSRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - every other failure is a refusal with its traceback
        pf = sys.modules.get("p07_rank_fuse")
        if pf is not None and isinstance(e, pf.ERRORS):
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
