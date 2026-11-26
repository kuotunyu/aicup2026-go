"""PREREG_08 competition 2: P1 probe features for the formal steps (docs/PREREG_08_STRENGTH_PROBE.md rev 1, sections
1.3-1.6 and 2.2; results/prereg08/design/spec.md sections A, B, C).

    CPU  PY=.venv/Scripts/python.exe      GPU  GPY=.venv-gpu/Scripts/python.exe     (Git Bash, PYTHONDONTWRITEBYTECODE=1)
    PY  tools/p08_p1.py manifest [--precommit] [--rerun --reason T]  -> results/prereg08/feat/inner_manifest.json
    GPY tools/p08_p1.py val                                           -> data/prereg08/val/chunk_*.npz, then
                                                                         results/prereg08/feat/val_p1.npz,
                                                                         val_seam_feats.npz, val_report.json
    PY  tools/p08_p1.py status [--out-dir D]                          (read only: progress, STOP file, estimate)
    smoke (never the validation file): PY tools/p08_p1.py val --smoke --questions Q.csv --out-dir data/prereg08/tmp/<x>
                                       [--max-games N] [--fake-scorer] [--workers W] [--ignore-other-gpu-jobs]

manifest  Only when the P2 probe has all 600 chunks or after 2026-10-08 23:59:59 (+08:00) (PREREG_08 1.4). The P1
          chunks data/prereg08/p1/{probe,bank,mono} (150 / 60 / 307) and the P2 chunks data/prereg08/p2/probe (600):
          per chunk SHA-256, games, failed, no target move, refs equal to the sets.json segment (p1_extract.set_games),
          metadata (format, sides, precision, device, batch, channels_last, neutral-metadata SHA, KataGo commit,
          p1_lib / p1_extract SHA, torch). The full checkpoint SHA-256; third_party/katago against SOURCE.txt.
          p2 = {complete, chunks, games, failed, used, reason}: used iff 600 / 600 valid chunks and failed <= 600.
          disjoint: probe players vs inner and validation players (players_split.csv), P2 vs P1 probe games.
          duplicates: probe (P1, P2) games whose move sequence (original orientation) equals a bank / mono game's
          (tune excludes them). move_cross_check: bank / mono P1 n_moves and target moves against the PREREG_07 pilot
          HS chunks and results/prereg04/opp/inner_o4.npz (both sides not failed: mismatches must be 0).
          Reads chunk metadata, counts, SHA-256 and the training SGFs only; computes no score; never reads data/local.
val       Only after results/prereg08/rank/frozen_config.json (gates passed, score_pending_val) and p1_map.npz are
          committed and clean, the design records (manifest, check, selfcheck, tune) committed and equal to what
          frozen_config.json records, the tools equal to tune's, no val_report.json yet, no other compute process on the
          GPU, and .venv-gpu's python / torch / cuDNN / GPU equal to frozen_config.json's reference. Then, in this
          order: read data/local/rank_val_questions.csv (question_id / num_games / color_i / sgf_i,
          p04_opp.read_question_slots) and val_keys.npz (p02 load_val_keys), check them row by row (1,589 questions,
          19,156 games); hash the validation move sequences and compare with the probe (record only); only then import
          p1_lib / p1_extract (their guard closes data/local for the rest of the process). Games in val_keys order;
          target colour = the question file's colour; p1_extract.run_chunk with ProbeScorer(fp16, cuda, 1,024,
          channels_last) per 200 games, written atomically to data/prereg08/val/chunk_<i>.npz. Rerunning resumes
          (chunks that pass the check are skipped; a chunk with other fields or another environment is refused, never
          overwritten). STOP file data/prereg08/val/STOP or --stop-after-min: end after the chunk in progress. When
          every chunk is there: val_p1.npz, val_seam_feats.npz (the first 400 games' means) and val_report.json.
          Failed games > 0 -> status stop, exit 1. Start it as a detached background process.
Smoke     a question file outside data/local (synthetic SGFs or training games); output under data/prereg08/tmp/;
          --fake-scorer replaces the network with a deterministic CPU function (tests).
Exit codes: 0 done (or stopped early on request); 1 status stop; 2 refused or error.
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

GPU_NAME = "NVIDIA GeForce RTX 4090"
SEAM_GAMES = 400
POSITIONS_PER_GAME = 228          # PREREG_08 1.6 estimate (status only)


def R():
    """tools/p08_rank_fuse.py (constants and helpers; imported lazily to keep the spawn children light)."""
    import p08_rank_fuse as r
    return r


class Refused(RuntimeError):
    pass


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ================================================================ pure helpers (tests)
def move_hash(sgf):
    """SHA-256 of the SGF's move sequence in its original orientation (p04_opp.encode_one); None without a move."""
    import p04_opp
    m = p04_opp.encode_one(sgf)
    return hashlib.sha256(m.tobytes()).hexdigest() if len(m) else None


def duplicate_refs(refs, hashes, other):
    """refs whose hash is in the set other (None never matches) -> sorted list of ints."""
    other = {h for h in other if h is not None}
    return sorted(int(r) for r, h in zip(refs, hashes) if h is not None and h in other)


def chunk_plan(n, size):
    return [(ci, s, min(n, s + size)) for ci, s in enumerate(range(0, n, size))]


class FakeProbeScorer:
    """Deterministic CPU stand-in for p1_lib.ProbeScorer (smoke tests only): float32 [T, 770] from the move index and
    the legal count of each position."""

    def position_features(self, feats):
        T = len(feats["move"])
        out = np.zeros((T, 770), dtype=np.float32)
        if T:
            nleg = feats["legal"].sum(1).astype(np.float64)
            mv = feats["move"].astype(np.float64)
            j = np.arange(384, dtype=np.float64)
            out[:, :384] = np.sin((mv[:, None] + 1.0) * (j + 1.0) * 1e-3)
            out[:, 384:768] = np.cos(nleg[:, None] * (j + 1.0) * 1e-4)
            out[:, 768] = -np.log(nleg)
            out[:, 769] = 0.9 * np.log(nleg)
        return out

    def close(self):
        pass


# ================================================================ manifest (spec C)
def p2_presence(r):
    d = r.SETS["p2_probe"]["dir"]
    n = r.SETS["p2_probe"]["chunks"]
    present = sum(1 for ci in range(n) if os.path.exists(os.path.join(d, f"chunk_{ci:05d}.npz")))
    return present, n


def scan_set(r, s, games):
    """Chunk records of one set against its sets.json games -> (set record, per-game arrays of the present chunks)."""
    spec = r.SETS[s]
    refs_all = np.array([g["ref"] for g in games], dtype=np.int64)
    plan = chunk_plan(len(games), r.RUN["chunk"])
    rows, arrays = [], []
    tot = {"games": 0, "failed": 0, "no_target_moves": 0, "no_move": 0, "failed_or_no_move": 0}
    refs_ok = meta_ok = True
    for ci, st, en in plan:
        p = os.path.join(spec["dir"], f"chunk_{ci:05d}.npz")
        if not os.path.exists(p):
            continue
        rec, ref, failed, cnt, nm = r.chunk_meta(p)
        rec["ref_matches_sets"] = bool(np.array_equal(ref, refs_all[st:en]))
        rec["meta_problems"] = r.chunk_meta_problems(rec, spec["key"])
        refs_ok = refs_ok and rec["ref_matches_sets"]
        meta_ok = meta_ok and not rec["meta_problems"]
        for k in tot:
            tot[k] += rec[k]
        rows.append(rec)
        arrays.append((ref, failed, cnt, nm))
    names = {f"chunk_{ci:05d}.npz" for ci, _s, _e in plan}
    extra = sorted(r.rel(f) for f in r.chunk_files(spec["dir"]) if os.path.basename(f) not in names)
    srec = {"chunks": rows, "chunks_expected": len(plan), "chunks_present": len(rows), "games_planned": len(games),
            "registered_chunks": spec["chunks"], "registered_games": spec["games"], "totals": tot,
            "refs_match_sets": bool(refs_ok), "meta_matches": bool(meta_ok), "extra_files": extra,
            "complete": bool(len(rows) == len(plan) == spec["chunks"] and len(games) == spec["games"] and refs_ok
                             and meta_ok and not extra)}
    return srec, arrays


def cmd_manifest(a):
    t0 = time.time()
    r = R()
    out = r.MANIFEST
    if os.path.exists(out):
        if not (a.rerun and a.reason):
            raise Refused(f"{r.rel(out)} exists; rerun only with --rerun --reason TEXT")
        if r.git(["ls-files", "--error-unmatch", "--", r.rel(out)])[0] == 0:
            raise Refused("inner_manifest.json is committed; it is never overwritten")
    ts = r.tool_state()
    r.require_tools(ts, a.precommit)
    pins = r.check_pins()
    now = datetime.datetime.now(datetime.timezone.utc)
    present, expected = p2_presence(r)
    if present < expected and now <= r.P2_CUTOFF:            # early refusal (no chunk is read)
        r.p2_decision(present, expected, False, 0, now)
    L1, F, E = r.import_p1_extract()
    sets1 = E.load_sets(r.P1_SETS)
    sets2 = E.load_sets(r.P2_SETS)
    print("checkpoint: full SHA-256 ...", flush=True)
    ck = r.P7.checkpoint_record()
    kprobs, krec = r.P7.katago_sources()
    sets_rec, games_of, arrays_of, torch_vals = {}, {}, {}, set()
    for s, sj in (("p1_probe", sets1), ("bank", sets1), ("mono", sets1), ("p2_probe", sets2)):
        t = time.time()
        games = E.set_games(sj, r.SETS[s]["key"])
        srec, arrays = scan_set(r, s, games)
        torch_vals |= {c["torch"] for c in srec["chunks"]}
        sets_rec[s] = srec
        games_of[s] = games
        arrays_of[s] = arrays
        print(f"{s}: {srec['chunks_present']}/{srec['chunks_expected']} chunks, {srec['totals']} "
              f"({time.time() - t:.0f}s)", flush=True)
    p2 = sets_rec["p2_probe"]
    p2rec = r.p2_decision(p2["chunks_present"], expected,
                          bool(p2["refs_match_sets"] and p2["meta_matches"] and not p2["extra_files"]
                               and p2["games_planned"] == r.SETS["p2_probe"]["games"]),
                          p2["totals"]["failed"], now)
    p2rec["games"] = p2["totals"]["games"]
    p2rec["no_target_moves_record_only"] = p2["totals"]["no_target_moves"]
    # disjoint (PREREG_08 2.2 item 6)
    import c10data
    split = c10data.read_split(r.SPLIT_CSV)
    inner_p = c10data.inner_players(split)
    val_p = {p for p, info in split.items() if info["split"] != "train"}
    pl1 = {g["player_id"] for g in sets1["probe"]["games"]}
    pl2 = {g["player_id"] for g in sets2["probe"]["games"]}
    g1 = {(g["rank"], g["game_id"]) for g in sets1["probe"]["games"]}
    g2 = {(g["rank"], g["game_id"]) for g in sets2["probe"]["games"]}
    dj = {"probe_players_p1": len(pl1), "probe_players_p2": len(pl2),
          "inner_overlap": len((pl1 | pl2) & inner_p), "validation_overlap": len((pl1 | pl2) & val_p),
          "p2_p1_game_overlap": len(g1 & g2)}
    dj["passed"] = dj["inner_overlap"] == 0 and dj["validation_overlap"] == 0 and dj["p2_p1_game_overlap"] == 0
    # duplicates: move sequences, original orientation (PREREG_08 1.4)
    t = time.time()
    hashes = {s: [move_hash(g["sgf_content"]) for g in games_of[s]] for s in games_of}
    other = set(hashes["bank"]) | set(hashes["mono"])
    dups = {s: duplicate_refs([g["ref"] for g in games_of[s]], hashes[s], other) for s in ("p1_probe", "p2_probe")}
    allp = [h for h in hashes["p1_probe"] + hashes["p2_probe"] if h is not None]
    dup_rec = {**dups, "count": {s: len(v) for s, v in dups.items()},
               "probe_games_without_moves": {s: sum(1 for h in hashes[s] if h is None) for s in ("p1_probe",
                                                                                                  "p2_probe")},
               "within_probe_repeats_record_only": len(allp) - len(set(allp)),
               "method": "sha256 of p04_opp.encode_one(sgf) (move sequence, original orientation); probe refs whose "
                         "hash equals a bank / mono game's are excluded by tune",
               "seconds": round(time.time() - t, 1)}
    del hashes, games_of
    # move cross-check (PREREG_08 2.2 item 5): P1 bank / mono vs the 07 pilot HS chunks and inner_o4.npz
    with np.load(r.P7.INNER_KEYS, allow_pickle=False) as z:
        keys = {k: np.asarray(z[k]).astype(str) for k in ("rank", "game_id", "player_id", "color")}
    n_inner = len(keys["rank"])
    p1_tot, p1_tgt = np.zeros(n_inner, np.int64), np.zeros(n_inner, np.int64)
    p1_fail = np.ones(n_inner, dtype=bool)
    rows_in = []
    for s in ("bank", "mono"):
        for ref, failed, cnt, nm in arrays_of[s]:
            p1_tot[ref], p1_tgt[ref], p1_fail[ref] = nm, cnt, failed
            rows_in.append(ref)
    rows_in = np.concatenate(rows_in) if rows_in else np.zeros(0, np.int64)
    hs_tot, hs_tgt = np.zeros(n_inner, np.int64), np.zeros(n_inner, np.int64)
    hs_fail = np.ones(n_inner, dtype=bool)
    for s in ("bank", "mono"):
        for f in sorted(glob.glob(os.path.join(r.PILOT_DATA, s, "chunk_*.npz"))):
            if ".tmp" in f:
                continue
            with np.load(f, allow_pickle=False) as z:
                ref = np.asarray(z["ref"]).astype(np.int64)
                hs_tot[ref] = np.asarray(z["n_moves"])
                hs_tgt[ref] = np.asarray(z["counts"])[:, 0]
                hs_fail[ref] = np.asarray(z["failed"], dtype=bool)
    with np.load(r.P7.INNER_O4, allow_pickle=False) as z:
        o4 = {k: np.asarray(z[k]) for k in ("rank", "game_id", "player_id", "color", "n_moves", "anomalies")}
    key_diff = [k for k in ("rank", "game_id", "player_id", "color") if not np.array_equal(o4[k].astype(str), keys[k])]
    ids = [f"{keys['rank'][i]}/{keys['game_id'][i]}" for i in rows_in]
    xc_hs = r.P7.move_cross_check(p1_tot[rows_in], p1_tgt[rows_in], p1_fail[rows_in], hs_tot[rows_in],
                                  hs_tgt[rows_in], hs_fail[rows_in], ids)
    nmo = o4["n_moves"][rows_in]
    xc_o4 = r.P7.move_cross_check(p1_tot[rows_in], p1_tgt[rows_in], p1_fail[rows_in], nmo.sum(axis=(1, 2)),
                                  nmo[:, 0, :].sum(1), o4["anomalies"][rows_in] > 0, ids)
    if key_diff:
        xc_o4["mismatches"] = None
    xc = {"vs_pilot_hs": {**xc_hs, "against": "data/prereg07/pilot/{bank,mono} chunks (n_moves, counts[:, 0])"},
          "vs_inner_o4": {**xc_o4, "against": r.rel(r.P7.INNER_O4), "inner_o4_sha256": r.sha_raw(r.P7.INNER_O4),
                          "key_arrays_differing": key_diff},
          "games": int(len(rows_in)),
          "definition": "both sides not failed: P1 n_moves == other total moves and P1 counts[:, 0] == other target "
                        "moves (HS counts[:, 0]; O4 sum n_moves[:, 0, :]); O4 failed = anomalies > 0"}
    complete = all(sets_rec[s]["complete"] for s in r.P1_SET_NAMES)
    man = {"format": r.FORMATS["manifest"], "what": "PREREG_08 2.2 / spec C: P1 / P2 chunk manifest (no score)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), "precommit": bool(a.precommit),
           "complete": bool(complete), "p2": p2rec, "sets": sets_rec, "torch_values": sorted(torch_vals),
           "neutral_metadata_sha256": L1.neutral_table_sha(), "checkpoint": ck,
           "katago": {**krec, "problems": kprobs}, "disjoint": dj, "duplicates": dup_rec, "move_cross_check": xc,
           "data_pins_sha256": {k: pins[k] for k in r.DATA_PINS}, "pinned_sha256": pins, "tool": ts,
           "environment": r.environment_cpu()}
    r.write_json(out, man)
    print(f"manifest: complete {complete}; P2 used {p2rec['used']} ({p2rec['reason']}); duplicates "
          f"{dup_rec['count']}; cross-check mismatches HS {xc_hs['mismatches']} O4 {xc_o4['mismatches']} -> "
          f"{r.rel(out)}")
    return 0


# ================================================================ val (spec C)
def gpu_env(fake=False):
    if fake:
        return {"python": sys.version.split()[0], "torch": "fake", "cudnn": "fake", "gpu": "fake", "driver": "fake",
                "numpy": np.__version__}
    import torch
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.cuda.is_available(), "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "driver": R().P7.nvidia_query().get("driver")}


def gpu_compute_processes():
    import p07_hs
    return p07_hs.gpu_compute_processes()


def chunk_state(path, refs, keys_sha, env, mode):
    """-> 'ok' (reuse), or raises when a chunk exists but cannot be reused (never overwritten)."""
    r = R()
    try:
        with np.load(path, allow_pickle=False) as z:
            got = {"format": str(z["format"]), "sides": [str(x) for x in z["sides"]],
                   "keys_sha256": str(z["keys_sha256"]), "meta_sha256": str(z["meta_sha256"]), "mode": str(z["mode"]),
                   "torch": str(z["torch"]), "cudnn": str(z["cudnn"]), "gpu_name": str(z["gpu_name"]),
                   "ckpt_sha256": str(z["ckpt_sha256"]), "means_shape": list(z["means"].shape)}
            same_refs = np.array_equal(np.asarray(z["ref"]), refs)
    except Exception as e:  # noqa: BLE001
        raise Refused(f"{path}: unreadable chunk ({e}); move it away and rerun")
    want = {"format": r.FORMATS["val_chunk"], "sides": r.RUN["sides"], "keys_sha256": keys_sha,
            "meta_sha256": r.NEUTRAL_META_SHA, "mode": mode, "means_shape": [len(refs), 2, r.F_DIM]}
    diff = [k for k in want if got[k] != want[k]] + ([] if same_refs else ["ref"])
    if diff:
        raise Refused(f"{path}: an existing chunk differs from this run ({diff}); never overwritten")
    envd = [k for k, e in (("torch", "torch"), ("cudnn", "cudnn"), ("gpu_name", "gpu")) if got[k] != str(env[e])]
    if envd:
        raise Refused(f"{path}: written with another environment ({envd}: {[got[k] for k in envd]}); stop and find "
                      f"out why")
    if got["ckpt_sha256"] not in (r.CKPT_SHA, "fake"):
        raise Refused(f"{path}: chunk without the full checkpoint SHA-256")
    return "ok"


def val_prerequisites(r):
    """Real mode, before data/local is opened -> (frozen config, records)."""
    fc, rec = r.frozen_prerequisites()
    if os.path.exists(os.path.join(r.FEAT, "val_report.json")):
        raise Refused("val_report.json exists: the registered validation P1 runs once (resume, never rerun)")
    procs, _raw = gpu_compute_processes()
    if procs is None or procs:
        raise Refused(f"nvidia-smi lists another compute process (or failed): {procs}")
    env = gpu_env()
    ref = fc["environment"]["venv_gpu_reference"]
    bad = [k for k in ("python", "torch", "cudnn", "gpu") if str(env[k]) != str(ref[k])]
    if bad:
        raise Refused(f".venv-gpu differs from frozen_config.json ({bad}: {[env[k] for k in bad]} vs "
                      f"{[ref[k] for k in bad]})")
    if env["gpu"] != GPU_NAME:
        raise Refused(f"GPU {env['gpu']} is not {GPU_NAME}")
    fpath = os.path.join(r.OUT, "frozen_config.json")
    rec.update(mode="real", environment=env, frozen_config_sha256=r.sha_lf(fpath),
               nvidia_compute_processes_before=procs)
    return fc, rec


def probe_overlap(r, fc, val_hashes):
    """Record only (PREREG_08 1.6): validation games whose move sequence equals a probe game's (expected 0). Reads
    the training SGFs (c10data, no data/local) before p1_lib is imported."""
    import c10data
    t = time.time()
    use_p2 = (fc.get("p1") or {}).get("probe_used") == "P1+P2"
    want = set()
    for path in ([r.P1_SETS, r.P2_SETS] if use_p2 else [r.P1_SETS]):
        with open(path, encoding="utf-8") as f:
            want |= {(g["rank"], g["game_id"]) for g in json.load(f)["probe"]["games"]}
    rows = c10data.load_rows(os.path.join(REPO, "data", "training"), r.SPLIT_CSV, part="fit", local_dir=None,
                             check_counts=True)
    probe_h = {move_hash(x["sgf_content"]) for x in rows if (x["rank"], x["game_id"]) in want}
    del rows
    probe_h.discard(None)
    hits = [i for i, h in enumerate(val_hashes) if h is not None and h in probe_h]
    return {"probe_used": "P1+P2" if use_p2 else "P1", "probe_games_hashed": len(want),
            "validation_games": len(val_hashes), "duplicates": len(hits), "first_rows": hits[:20],
            "seconds": round(time.time() - t, 1), "note": "record only; validation players are not fit players"}


def cmd_val(a):
    t0 = time.time()
    r = R()
    if a.smoke:
        if not a.questions or not a.out_dir:
            raise Refused("--smoke needs --questions (outside data/local) and --out-dir (under data/prereg08/tmp)")
        qfile = os.path.abspath(a.questions)
        if r.under_data_local(qfile):
            raise Refused("smoke runs never read data/local; a question file there needs the real mode")
        out_dir = os.path.abspath(a.out_dir)
        if not r.inside(out_dir, r.TMP):
            raise Refused(f"--smoke --out-dir must be under {r.rel(r.TMP)}")
        chunk_dir, feat_dir = os.path.join(out_dir, "val"), os.path.join(out_dir, "feat")
        if os.path.exists(os.path.join(feat_dir, "val_report.json")):
            raise Refused("smoke val_report.json exists; use a fresh --out-dir")
        pre, fc, mode = {"mode": "smoke"}, None, "smoke"
        env = gpu_env(fake=a.fake_scorer)
        if not a.fake_scorer and not a.ignore_other_gpu_jobs:
            procs, _ = gpu_compute_processes()
            if procs:
                raise Refused(f"nvidia-smi lists another compute process: {procs}")
    else:
        if a.questions or a.out_dir or a.max_games or a.fake_scorer or a.ignore_other_gpu_jobs:
            raise Refused("real mode takes no --questions / --out-dir / --max-games / --fake-scorer / "
                          "--ignore-other-gpu-jobs")
        fc, pre = val_prerequisites(r)
        env, mode = pre["environment"], "real"
        qfile = r.P7.QUESTIONS
        chunk_dir, feat_dir = r.VAL_DATA, r.FEAT
    r.refuse_label_path(qfile)
    # ---- 1 the question file and the keys (before p1_lib / p1_extract)
    import p04_opp
    import p02_rank_fuse as rf
    q_sha = r.sha_raw(qfile)
    slots = p04_opp.read_question_slots(qfile)
    if a.smoke:
        keys_sha = q_sha
        vk = {"question_id": [s[0] for s in slots], "slot": [s[1] for s in slots], "color": [s[2] for s in slots],
              "n": len(slots)}
    else:
        if q_sha != r.P7.QUESTIONS_SHA:
            raise Refused(f"question file SHA-256 {q_sha} != {r.P7.QUESTIONS_SHA}")
        vk = rf.load_val_keys(os.path.dirname(r.P7.VAL_KEYS))
        if vk["sha256"] != r.P7.VAL_KEYS_SHA:
            raise Refused("val_keys.npz is not the registered file")
        keys_sha = vk["sha256"]
    p04_opp.slots_match_keys(slots, vk)
    n_q = len(dict.fromkeys(vk["question_id"]))
    if not a.smoke and (n_q != r.VAL_QUESTIONS or vk["n"] != r.VAL_GAMES):
        raise Refused(f"{n_q} questions / {vk['n']} games, registered {r.VAL_QUESTIONS} / {r.VAL_GAMES}")
    games = [{"sgf_content": sgf, "color": col, "ref": i} for i, (_q, _s, col, sgf) in enumerate(slots)]
    if a.max_games:
        games = games[:a.max_games]
    val_hashes = [move_hash(g["sgf_content"]) for g in games]
    del slots
    overlap = probe_overlap(r, fc, val_hashes) if not a.smoke else {"skipped": "smoke"}
    # ---- 2 only now p1_lib / p1_extract (their guard closes data/local for this process)
    L1, _F, E = r.import_p1_extract()
    import multiprocessing as mp
    os.makedirs(chunk_dir, exist_ok=True)
    refs_all = np.array([g["ref"] for g in games], dtype=np.int64)
    chunks = chunk_plan(len(games), r.RUN["chunk"])
    todo = []
    for ci, s, e in chunks:
        p = os.path.join(chunk_dir, f"chunk_{ci:05d}.npz")
        if os.path.exists(p):
            chunk_state(p, refs_all[s:e], keys_sha, env, mode)
        else:
            todo.append((ci, s, e))
    print(json.dumps({"time_utc": now_iso(), "msg": "val start", "mode": mode, "games": len(games),
                      "chunks": len(chunks), "todo": len(todo)}), flush=True)
    stopped = None
    tool_sha = r.sha_lf(os.path.abspath(__file__))
    shas = json.dumps(L1.code_shas(), sort_keys=True)
    if todo:
        if a.fake_scorer:
            sc, ck = FakeProbeScorer(), {"bytes": r.CKPT_BYTES, "sha256": "fake"}
            torch_v, precision, device = "fake", "fake-fp32", "cpu"
        else:
            import torch
            if not torch.cuda.is_available():
                raise Refused("CUDA is not available")
            if env["gpu"] != GPU_NAME:
                raise Refused(f"GPU {env['gpu']} is not {GPU_NAME}")
            torch.backends.cudnn.benchmark = r.RUN["cudnn_benchmark"]
            ck = L1.H.check_checkpoint(L1.H.CKPT_PATH, full_hash=True)
            if ck.get("sha256") != r.CKPT_SHA:
                raise Refused("checkpoint SHA-256 differs")
            model = L1.load_model(device=r.RUN["device"])
            sc = L1.ProbeScorer(model, device=r.RUN["device"], precision=r.RUN["precision"], batch=r.RUN["batch"],
                                channels_last=r.RUN["channels_last"])
            torch_v, precision, device = torch.__version__, r.RUN["precision"], r.RUN["device"]
        group_positions = r.RUN["batch"] * r.RUN["group_batches"]
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
                res, t_gpu = E.run_chunk(pool, sc, games[s:e], group_positions)
                secs = time.time() - t1
                npos = int(len(res["pos_side"]))
                path = os.path.join(chunk_dir, f"chunk_{ci:05d}.npz")
                tmp = path + ".tmp.npz"
                np.savez(tmp, format=np.array(r.FORMATS["val_chunk"]), set=np.array("val" if mode == "real" else
                                                                                       "val_smoke"),
                         sides=np.array(r.RUN["sides"]), ref=refs_all[s:e], precision=np.array(precision),
                         batch=np.array(r.RUN["batch"]), channels_last=np.array(r.RUN["channels_last"]),
                         device=np.array(device), torch=np.array(torch_v), cudnn=np.array(str(env["cudnn"])),
                         gpu_name=np.array(str(env["gpu"])), driver=np.array(str(env["driver"])),
                         ckpt_bytes=np.array(ck["bytes"]), ckpt_sha256=np.array(ck["sha256"]),
                         katago_commit=np.array(r.KATAGO_COMMIT), meta_sha256=np.array(L1.neutral_table_sha()),
                         code_sha256=np.array(shas), keys_sha256=np.array(keys_sha),
                         questions_sha256=np.array(q_sha), tool_sha256=np.array(tool_sha), mode=np.array(mode),
                         created_utc=np.array(now_iso()), seconds=np.array(secs), gpu_seconds=np.array(t_gpu), **res)
                os.replace(tmp, path)
                el = time.time() - t_run
                print(json.dumps({"time_utc": now_iso(), "msg": "chunk done", "chunk": ci, "games": e - s,
                                  "failed": int(res["failed"].sum()), "positions": npos, "seconds": round(secs, 1),
                                  "positions_per_s": round(npos / max(secs, 1e-9), 1),
                                  "progress": f"{k + 1}/{len(todo)}",
                                  "eta_min": round(el / (k + 1) * (len(todo) - k - 1) / 60, 1)}), flush=True)
        sc.close()
    if stopped:
        return 0
    return assemble(r, chunk_dir, feat_dir, chunks, vk, keys_sha, q_sha, env, pre, fc, overlap, mode, t0)


def assemble(r, chunk_dir, feat_dir, chunks, vk, keys_sha, q_sha, env, pre, fc, overlap, mode, t0):
    rows, parts = [], []
    for ci, s, e in chunks:
        p = os.path.join(chunk_dir, f"chunk_{ci:05d}.npz")
        with np.load(p, allow_pickle=False) as z:
            d = {k: np.asarray(z[k]) for k in ("ref", "means", "counts", "n_moves", "failed", "pos_side")}
            d.update(created_utc=str(z["created_utc"]), seconds=float(z["seconds"]), errors=str(z["errors"]))
        parts.append(d)
        failed = d["failed"].astype(bool)
        cnt = d["counts"][:, 0]
        rows.append({"file": r.rel(p), "sha256": r.sha_raw(p), "games": int(len(d["ref"])),
                     "failed": int(failed.sum()), "no_target_moves": int(((cnt == 0) & ~failed).sum()),
                     "no_move": int((d["n_moves"] == 0).sum()), "created_utc": d["created_utc"],
                     "seconds": d["seconds"], "positions": int(len(d["pos_side"])), "errors": json.loads(d["errors"])})
    ref = np.concatenate([d["ref"] for d in parts])
    n = len(ref)
    if not np.array_equal(ref, np.arange(n)):
        raise Refused("chunk refs are not the val_keys rows in order")
    means = np.concatenate([d["means"] for d in parts]).astype(np.float32)
    counts = np.concatenate([d["counts"] for d in parts]).astype(np.int64)
    n_moves = np.concatenate([d["n_moves"] for d in parts]).astype(np.int64)
    failed = np.concatenate([d["failed"] for d in parts]).astype(bool)
    no_target = (counts[:, 0] == 0) & ~failed
    scal = means[:, :, 768:770].astype(np.float64)               # [G, side, (log p, entropy)] means
    os.makedirs(feat_dir, exist_ok=True)
    vp = os.path.join(feat_dir, "val_p1.npz")
    np.savez(vp + ".tmp.npz", question_id=np.array(vk["question_id"][:n]),
             slot=np.array(vk["slot"][:n], dtype=np.int64), color=np.array(vk["color"][:n]), counts=counts,
             n_moves=n_moves, failed=failed, no_target_moves=no_target, scal=scal, ref=ref,
             scal_fields=np.array(["logp", "entropy"]), sides=np.array(r.RUN["sides"]))
    os.replace(vp + ".tmp.npz", vp)
    k = min(SEAM_GAMES, n)
    seam_chunks = [row for row, (ci, s, e) in zip(rows, chunks) if s < k]
    sm = os.path.join(feat_dir, "val_seam_feats.npz")
    np.savez(sm + ".tmp.npz", means=means[:k], counts=counts[:k], n_moves=n_moves[:k], failed=failed[:k],
             games=np.array(k), chunk_sha256=np.array([row["sha256"] for row in seam_chunks]),
             question_id=np.array(vk["question_id"][:k]), slot=np.array(vk["slot"][:k], dtype=np.int64),
             color=np.array(vk["color"][:k]))
    os.replace(sm + ".tmp.npz", sm)
    fails = int((failed | (n_moves == 0)).sum())
    starts = [datetime.datetime.fromisoformat(row["created_utc"]) - datetime.timedelta(seconds=row["seconds"])
              for row in rows]
    pos = sum(row["positions"] for row in rows)
    secs = sum(row["seconds"] for row in rows)
    fpath = os.path.join(r.OUT, "frozen_config.json")
    rep = {"format": r.FORMATS["val_report"], "what": "PREREG_08 1.6 validation P1 features (no score)",
           "mode": mode, "created": now_iso(), "status": "ok" if fails == 0 else "stop",
           "games": n, "questions": len(dict.fromkeys(vk["question_id"][:n])), "failed": int(failed.sum()),
           "failed_or_no_move": fails, "no_target_moves": int(no_target.sum()), "chunks": rows,
           "first_chunk_started_utc": min(starts).isoformat(timespec="seconds"),
           "last_chunk_created_utc": max(row["created_utc"] for row in rows), "positions": pos,
           "seconds_in_chunks": round(secs, 1), "positions_per_s": round(pos / max(secs, 1e-9), 1),
           "environment": env, "frozen_config_sha256": (r.sha_lf(fpath) if fc is not None else None),
           "frozen_config_commit_time": pre.get("frozen_commit_time"), "prerequisites": pre,
           "probe_overlap_record_only": overlap, "questions_sha256": q_sha, "keys_sha256": keys_sha,
           "val_p1_sha256": r.sha_raw(vp), "val_seam_feats_sha256": r.sha_raw(sm), "run": r.RUN,
           "tool_sha256": r.sha_lf(os.path.abspath(__file__)), "seconds_this_call": round(time.time() - t0, 1),
           "next": "commit val_p1.npz, val_seam_feats.npz, val_report.json; then p08_rank_fuse.py apply"
           if fails == 0 else "failed games > 0: stop and find out why (PREREG_08 2.7)"}
    r.write_json(os.path.join(feat_dir, "val_report.json"), rep)
    print(f"val: {n} games, failed {fails}, {pos} positions in {secs:.0f} s -> {r.rel(feat_dir)}")
    return 0 if fails == 0 else 1


# ================================================================ status
def cmd_status(a):
    r = R()
    d = os.path.join(os.path.abspath(a.out_dir), "val") if a.out_dir else r.VAL_DATA
    files = sorted(f for f in glob.glob(os.path.join(d, "chunk_*.npz")) if ".tmp" not in f)
    secs, pos = 0.0, 0
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            secs += float(z["seconds"])
            pos += int(len(z["pos_side"]))
    total = (r.VAL_GAMES + r.RUN["chunk"] - 1) // r.RUN["chunk"]
    rate = pos / secs if secs else None
    left_games = max(0, r.VAL_GAMES - len(files) * r.RUN["chunk"])
    out = {"folder": r.rel(d) if r.inside(d, REPO) else d, "chunks_present": len(files), "chunks_total": total,
           "stop_file": os.path.exists(os.path.join(d, "STOP")), "positions_per_s": rate,
           "estimate_hours_left": (left_games * POSITIONS_PER_GAME / rate / 3600) if rate else None,
           "val_report": os.path.exists(os.path.join(r.FEAT, "val_report.json"))}
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
    v.add_argument("--workers", type=int, default=20)
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
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - every other refusal of the shared tools is a refusal too
        r = sys.modules.get("p08_rank_fuse")
        if (r is not None and isinstance(e, r.ERRORS)) or type(e).__name__ in ("P1Error", "HSError",
                                                                                 "LocalDataRefused", "P04Error"):
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
