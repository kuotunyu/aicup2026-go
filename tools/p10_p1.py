"""PREREG_10 (competition 1) part B: KataGo HumanSL neutral-probe (P1) features for the official-shape mock pools of
fit players (docs/PREREG_10_PLAYER_GUARD_P1.md rev 1, sections 2.1-2.2).

    Git Bash, repository root, PYTHONDONTWRITEBYTECODE=1:
    .venv/Scripts/python.exe     tools/p10_p1.py sets                 -> data/prereg10/mock/sets.json
                                                                         + results/prereg10/feat/mock_sets_record.json
    .venv-gpu/Scripts/python.exe tools/p10_p1.py extract --set mock   -> data/prereg10/mock/chunk_*.npz (resumable)
    .venv/Scripts/python.exe     tools/p10_p1.py status
    .venv/Scripts/python.exe     tools/p10_p1.py manifest --set mock  -> results/prereg10/feat/mock_manifest.json
                                                                         (only when every mock chunk is there)
    .venv-gpu/Scripts/python.exe tools/p10_p1.py extract --set val    -> data/prereg10/val/chunk_*.npz (resumable), then
                                                                         results/prereg10/feat/val_report.json and
                                                                         val_seam_feats.npz (PREREG_10 2.5)
    smoke (never data/local): .venv-gpu/Scripts/python.exe tools/p10_p1.py extract --set val --smoke
        --candidates C.csv --questions Q.csv --out-dir data/prereg10/tmp/<x> [--max-games N] [--fake-scorer]

Players: fit players (split v1) with >= 120 black games in the PREREG_06 fit feature store
(data/prereg06/feats/meta.npz). Per player, black games ordered by sha256('c10-prereg10|mock|<player_id>|<game_id>'):
the first 100 are candidate games, the next 20 the base question reserve; of the remaining games, those whose
canonical first-4-ply key (results/prereg10/explore/x0_open_keys.py, data/prereg10/tmp/open_keys.npz column c4) does
not occur among the player's 100 candidate games, in the same order, the first <= 20 are the nov4 question reserve.
Only fit players (tools/c10data.load_rows part 'fit', local_dir None); never data/local (the hs_lib guard is installed
when p1_lib is imported). Reuses results/prereg08/p1/p1_lib.py and p1_extract.run_chunk unchanged (imported).
No competition-2 feature values are read: only the shared extraction code.
Yield the GPU: create data/prereg10/mock/STOP (stops before the next chunk); delete it and run again to resume.
`extract` refuses to start while nvidia-smi lists another compute process.

extract --set val (added after 15b3e98; the sets / mock extraction code above is unchanged): only after
results/prereg10/player/frozen_config.json (B passed, not a preview) and p1_lda.npz are committed and clean, and no
val_report.json exists. In this order: install the p06 data/local guard (tools/p06_style.py); read only
data/local/player_val_candidates.csv and player_val_questions.csv (inside p06_style.allow_local(); their SHA-256 must
equal the inputs of results/prereg06/player/apply_report.json); build the game list (candidate games in file order,
then the question games in question / slot order; target colour = the row's color / color_i); only then import
p1_lib / p1_extract (hs_lib's guard closes data/local for the rest of the process). Chunks of 200 games in
data/prereg10/val/ (set 'p10val'; STOP file data/prereg10/val/STOP yields the GPU). When every chunk is there:
results/prereg10/feat/val_report.json (chunk SHA-256s, failed, no-target, keys SHA, inputs) and val_seam_feats.npz
(side-0 means, counts, failed of the first 400 games, for the test-day GPU seam of PREREG_10 4.2).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
P1_DIR = os.path.join(REPO, "results", "prereg08", "p1")
for p in (P1_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402

OUT = os.path.join(REPO, "data", "prereg10", "mock")
SETS = os.path.join(OUT, "sets.json")
RECORD = os.path.join(REPO, "results", "prereg10", "feat", "mock_sets_record.json")
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
META = os.path.join(REPO, "data", "prereg06", "feats", "meta.npz")
KEYS = os.path.join(REPO, "data", "prereg10", "tmp", "open_keys.npz")
SALT = "c10-prereg10|mock"
N_CAND, N_BASE, N_NOV, MIN_BLACK = 100, 20, 20, 120
CHUNK = 200
FORMAT = "c10-prereg10 mock sets v1"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_sets(_a):
    if os.path.exists(SETS):
        raise SystemExit("%s exists; not overwritten" % SETS)
    with np.load(META, allow_pickle=False) as z:
        rank, gid, pid, color = z["rank"], z["game_id"], z["player_id"], z["color"]
    with np.load(KEYS, allow_pickle=False) as z:
        c4 = z["c4"]
    assert len(c4) == len(gid), "open_keys rows differ from the feature store"
    from c10data import load_rows
    fit = load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    fit_key = {(r["rank"], r["game_id"]): (r["player_id"], r["color"]) for r in fit}
    del fit
    byb = {}
    for i in range(len(gid)):
        if color[i] == "B":
            byb.setdefault(str(pid[i]), []).append(i)
    players, games, n_nov = [], [], {}
    for p in sorted(byb):
        rows = byb[p]
        if len(rows) < MIN_BLACK:
            continue
        rows = sorted(rows, key=lambda r: hashlib.sha256(("%s|%s|%s" % (SALT, p, gid[r])).encode()).hexdigest())
        cand, base, rest = rows[:N_CAND], rows[N_CAND:N_CAND + N_BASE], rows[N_CAND + N_BASE:]
        ckeys = {str(c4[r]) for r in cand}
        nov = [r for r in rest if str(c4[r]) not in ckeys][:N_NOV]
        n_nov[p] = len(nov)
        players.append(p)
        for role, rr in (("cand", cand), ("base", base), ("nov4", nov)):
            for k, r in enumerate(rr):
                key = (str(rank[r]), str(gid[r]))
                fp = fit_key.get(key)
                if fp is None or fp[0] != p or fp[1] != "B":
                    raise SystemExit("store row %d is not this fit player's black game" % r)
                games.append({"rank": key[0], "game_id": key[1], "player_id": p, "color": "B", "role": role,
                              "k": k, "store_row": int(r)})
    nn = np.array(list(n_nov.values()))
    sets = {"format": FORMAT, "salt": SALT, "rule": "PREREG_10 rev 1 section 2.2",
            "min_black_games": MIN_BLACK, "n_cand": N_CAND, "n_base": N_BASE, "n_nov_max": N_NOV,
            "inputs_sha256": {"meta.npz": sha_file(META), "open_keys.npz": sha_file(KEYS)},
            "players": players, "games": games}
    os.makedirs(OUT, exist_ok=True)
    with open(SETS + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(sets, f)
    os.replace(SETS + ".tmp", SETS)
    rec = {"format": "c10-prereg10 mock sets record v1", "sets_json": "data/prereg10/mock/sets.json",
           "sets_sha256": sha_file(SETS), "players": len(players), "games": len(games),
           "by_role": {r: sum(1 for g in games if g["role"] == r) for r in ("cand", "base", "nov4")},
           "nov4_per_player": {"min": int(nn.min()), "mean": round(float(nn.mean()), 2), "zero": int((nn == 0).sum()),
                               "full_20": int((nn == N_NOV).sum())},
           "chunks": (len(games) + CHUNK - 1) // CHUNK, "inputs_sha256": sets["inputs_sha256"],
           "tool_sha256": sha_file(os.path.abspath(__file__))}
    os.makedirs(os.path.dirname(RECORD), exist_ok=True)
    with open(RECORD, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(json.dumps(rec, indent=1))
    return 0


def load_games():
    from c10data import load_rows
    with open(SETS, encoding="utf-8") as f:
        s = json.load(f)
    rows = load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    byk = {(r["rank"], r["game_id"]): r for r in rows}
    out = []
    for j, g in enumerate(s["games"]):
        r = byk[(g["rank"], g["game_id"])]
        if r["player_id"] != g["player_id"] or r["color"] != "B":
            raise SystemExit("mock game %d: player or colour differs" % j)
        out.append({**r, "ref": j})
    return out


def cmd_extract(a):
    if a.set == "val":
        return cmd_extract_val(a)
    if a.set != "mock":
        raise SystemExit("only --set mock is implemented here (val comes after the freeze, PREREG_10 2.4-2.5)")
    import torch  # noqa: F401
    import p1_lib as P  # pin check + data/local guard
    import p1_extract as PX
    H = P.H
    PX.CHUNK_ROOT = OUT
    games = load_games()
    if a.max_games:
        games = games[:a.max_games]
    refs = np.array([g["ref"] for g in games], np.int64)
    chunks = [(ci, s, min(len(games), s + CHUNK)) for ci, s in enumerate(range(0, len(games), CHUNK))]
    path = lambda ci: os.path.join(OUT, "chunk_%05d.npz" % ci)  # noqa: E731

    def ok(ci, s, e):
        try:
            with np.load(path(ci), allow_pickle=False) as z:
                return str(z["format"]) == P.CHUNK_FORMAT and np.array_equal(z["ref"], refs[s:e]) \
                    and str(z["meta_sha256"]) == P.neutral_table_sha() and str(z["set"]) == "p10mock"
        except Exception:
            return False
    todo = [c for c in chunks if not ok(*c)]
    PX.log("p10 mock extract start", games=len(games), chunks=len(chunks), todo=len(todo), workers=a.workers)
    if not todo:
        return 0
    procs, _ = H.other_compute_processes()
    if procs:
        raise SystemExit("another compute process on the GPU: %s" % procs)
    import torch
    torch.backends.cudnn.benchmark = True
    ck = H.check_checkpoint(H.CKPT_PATH, full_hash=False)
    model = P.load_model(device="cuda")
    sc = P.ProbeScorer(model, device="cuda", precision="fp16", batch=1024, channels_last=True)
    t_run = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, (ci, s, e) in enumerate(todo):
            if os.path.exists(os.path.join(OUT, "STOP")):
                PX.log("STOP file: stopping before the next chunk", chunks_left=len(todo) - k)
                break
            t0 = time.time()
            res, t_gpu = PX.run_chunk(pool, sc, games[s:e], 4096)
            secs = time.time() - t0
            tmp = path(ci) + ".tmp.npz"
            np.savez(tmp, format=np.array(P.CHUNK_FORMAT), set=np.array("p10mock"), sides=np.array(PX.SIDES),
                     ref=refs[s:e], precision=np.array("fp16"), meta_sha256=np.array(P.neutral_table_sha()),
                     ckpt_bytes=np.array(ck["bytes"]), code_sha256=np.array(json.dumps(P.code_shas(), sort_keys=True)),
                     sets_sha256=np.array(sha_file(SETS)), created_utc=np.array(H.utc_now()), seconds=np.array(secs),
                     **res)
            os.replace(tmp, path(ci))
            el = time.time() - t_run
            npos = int(len(res["pos_side"]))
            PX.log("chunk done", chunk=ci, games=e - s, failed=int(res["failed"].sum()), positions=npos,
                   seconds=round(secs, 1), positions_per_s=round(npos / secs, 1), gpu_busy_frac=round(t_gpu / secs, 3),
                   progress="%d/%d" % (k + 1, len(todo)), eta_min=round(el / (k + 1) * (len(todo) - k - 1) / 60, 1))
    sc.close()
    PX.log("p10 mock extract end", seconds=round(time.time() - t_run, 1))
    return 0


def cmd_status(_a):
    with open(SETS, encoding="utf-8") as f:
        n = len(json.load(f)["games"])
    have = [x for x in os.listdir(OUT) if x.startswith("chunk_") and x.endswith(".npz") and ".tmp" not in x]
    print(json.dumps({"games": n, "chunks": (n + CHUNK - 1) // CHUNK, "present": len(have),
                      "STOP": os.path.exists(os.path.join(OUT, "STOP"))}))
    return 0


# ================================================================ manifest of the mock chunks (added after 15b3e98)
MANIFEST = os.path.join(REPO, "results", "prereg10", "feat", "mock_manifest.json")


def cmd_manifest(a):
    """Every mock chunk present and matching sets.json -> results/prereg10/feat/mock_manifest.json (once)."""
    if a.set != "mock":
        raise SystemExit("manifest is for --set mock (the validation chunks are recorded by extract --set val)")
    import p10_fuse as F
    if os.path.exists(MANIFEST):
        raise SystemExit("%s exists; not overwritten" % MANIFEST)
    with open(SETS, encoding="utf-8") as f:
        n = len(json.load(f)["games"])
    sets_sha = sha_file(SETS)
    plan = [(ci, s, min(n, s + CHUNK)) for ci, s in enumerate(range(0, n, CHUNK))]
    rows, problems = [], []
    for ci, s, e in plan:
        p = os.path.join(OUT, "chunk_%05d.npz" % ci)
        if not os.path.exists(p):
            problems.append("chunk %d missing" % ci)
            continue
        r = F.chunk_record(p)
        with np.load(p, allow_pickle=False) as z:
            ref_ok = np.array_equal(np.asarray(z["ref"], np.int64), np.arange(s, e))
        if not ref_ok or r["set"] != "p10mock" or r.get("sets_sha256") != sets_sha or r["format"] != F.CHUNK_FORMAT:
            problems.append("chunk %d: refs / set / sets SHA / format differ" % ci)
        rows.append(r)
    if problems:
        print(json.dumps({"complete": False, "problems": problems[:20], "n_problems": len(problems)}, indent=1))
        return 1
    rec = {"format": F.FORMATS["manifest"], "set": "mock", "complete": True,
           "sets_json": "data/prereg10/mock/sets.json", "sets_sha256": sets_sha, "games": n, "chunks": rows,
           "totals": {"failed": sum(r["failed"] for r in rows),
                      "no_target_moves": sum(r["no_target_moves"] for r in rows)},
           "meta_sha256": sorted({r["meta_sha256"] for r in rows}),
           "code_sha256": sorted({r["code_sha256"] for r in rows}),
           "precision": sorted({r["precision"] for r in rows}), "ckpt_bytes": sorted({r["ckpt_bytes"] for r in rows}),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tool_sha256": sha_file(os.path.abspath(__file__))}
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(json.dumps({k: rec[k] for k in ("complete", "games", "totals", "meta_sha256")}, indent=1))
    return 0


# ============================================================== extract --set val (PREREG_10 2.5; added after 15b3e98)
VAL_OUT = os.path.join(REPO, "data", "prereg10", "val")
VAL_TMP = os.path.join(REPO, "data", "prereg10", "tmp")
FEAT_DIR = os.path.join(REPO, "results", "prereg10", "feat")
PLAYER_DIR = os.path.join(REPO, "results", "prereg10", "player")
P06_APPLY = os.path.join(REPO, "results", "prereg06", "player", "apply_report.json")
SEAM_GAMES = 400


class FakeProbeScorer:
    """Deterministic CPU stand-in for p1_lib.ProbeScorer (smoke runs only; as tools/p08_p1.py's)."""

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


def _git_clean(path):
    import subprocess
    r = os.path.relpath(path, REPO).replace(os.sep, "/")
    if subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", "--", r], capture_output=True).returncode:
        return False
    return subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", r]).returncode == 0


def val_prerequisites(F):
    """Real mode: the freeze committed (B passed), no val_report.json yet. -> (frozen config, its LF SHA-256)."""
    fp = os.path.join(PLAYER_DIR, "frozen_config.json")
    lp = os.path.join(PLAYER_DIR, "p1_lda.npz")
    for p in (fp, lp):
        if not os.path.exists(p) or not _git_clean(p):
            raise SystemExit("refused: %s is not committed and clean (the validation P1 comes after the freeze, "
                             "PREREG_10 2.4)" % os.path.relpath(p, REPO))
    import p10_apply as A
    fc = F.read_json(fp)
    A.frozen_usable(fc)
    if F.sha_raw(lp) != fc["lda"]["sha256"]:
        raise SystemExit("refused: p1_lda.npz differs from frozen_config.json")
    if os.path.exists(os.path.join(FEAT_DIR, "val_report.json")):
        raise SystemExit("refused: results/prereg10/feat/val_report.json exists")
    return fc, F.sha_lf(fp)


def cmd_extract_val(a):
    import p06_style as st
    st.install_guard()
    import p10_fuse as F
    import p10_apply as A
    if a.smoke:
        if not (a.candidates and a.questions and a.out_dir):
            raise SystemExit("--smoke needs --candidates, --questions (outside data/local) and --out-dir")
        cfile, qfile = os.path.abspath(a.candidates), os.path.abspath(a.questions)
        if st.under_local(cfile) or st.under_local(qfile):
            raise SystemExit("smoke runs never read data/local")
        root = os.path.abspath(a.out_dir)
        if not F.inside(root, VAL_TMP) or os.path.normcase(root) == os.path.normcase(VAL_TMP):
            raise SystemExit("--smoke --out-dir must be a folder under data/prereg10/tmp")
        chunk_dir, feat_dir, mode, set_name = os.path.join(root, "val"), root, "smoke", "p10val_smoke"
        fc_sha, want_in = None, None
    else:
        if a.candidates or a.questions or a.out_dir or a.max_games or a.fake_scorer:
            raise SystemExit("real mode takes no --candidates / --questions / --out-dir / --max-games / --fake-scorer")
        _fc, fc_sha = val_prerequisites(F)
        cfile, qfile = A.VAL_CANDIDATES, A.VAL_QUESTIONS
        chunk_dir, feat_dir, mode, set_name = VAL_OUT, FEAT_DIR, "real", A.VAL_SET
        want_in = F.read_json(P06_APPLY)["inputs"]
    if os.path.exists(os.path.join(feat_dir, "val_report.json")):
        raise SystemExit("val_report.json exists in %s" % feat_dir)
    # ---- 1 the two game files, before p1_lib / p1_extract
    import p02_player_fuse as pf
    with st.allow_local():
        in_sha = {"candidates_csv": sha_file(cfile), "questions_csv": sha_file(qfile)}
        cand_rows = pf.read_candidates(cfile)
        q_rows = pf.read_questions(qfile)
    if want_in is not None and any(in_sha[k] != want_in.get(k) for k in in_sha):
        raise SystemExit("refused: the validation game files differ from the ones 06 applied to")
    keys, cols = A.val_keys(cand_rows, q_rows)
    sgfs = [r[3] for r in cand_rows] + [x for _q, g in q_rows for _s, _c, x in g]
    del cand_rows, q_rows
    if any(c not in ("B", "W") for c in cols):
        raise SystemExit("a game colour is not B or W")
    ksha = A.keys_sha(keys)
    games = [{"sgf_content": x, "color": c, "ref": j} for j, (x, c) in enumerate(zip(sgfs, cols))]
    del sgfs
    if a.max_games:
        games = games[:a.max_games]
    refs = np.array([g["ref"] for g in games], np.int64)
    chunks = [(ci, s, min(len(games), s + CHUNK)) for ci, s in enumerate(range(0, len(games), CHUNK))]
    # ---- 2 only now p1_lib / p1_extract (hs_lib's guard closes data/local)
    import p1_lib as P
    import p1_extract as PX
    H = P.H
    PX.CHUNK_ROOT = chunk_dir
    os.makedirs(chunk_dir, exist_ok=True)
    path = lambda ci: os.path.join(chunk_dir, "chunk_%05d.npz" % ci)  # noqa: E731

    def ok(ci, s, e):
        try:
            with np.load(path(ci), allow_pickle=False) as z:
                return str(z["format"]) == P.CHUNK_FORMAT and np.array_equal(z["ref"], refs[s:e]) \
                    and str(z["meta_sha256"]) == P.neutral_table_sha() and str(z["set"]) == set_name \
                    and str(z["keys_sha256"]) == ksha
        except Exception:
            return False
    todo = []
    for ci, s, e in chunks:
        if not ok(ci, s, e):
            if os.path.exists(path(ci)):
                raise SystemExit("chunk %d exists but does not match (never overwritten)" % ci)
            todo.append((ci, s, e))
    PX.log("p10 val extract start", mode=mode, games=len(games), chunks=len(chunks), todo=len(todo))
    stopped = None
    if todo:
        if a.fake_scorer:
            sc, ck, prec = FakeProbeScorer(), {"bytes": 0}, "fake-fp32"
        else:
            procs, _ = H.other_compute_processes()
            if procs:
                raise SystemExit("another compute process on the GPU: %s" % procs)
            import torch
            torch.backends.cudnn.benchmark = True
            ck = H.check_checkpoint(H.CKPT_PATH, full_hash=False)
            model = P.load_model(device="cuda")
            sc = P.ProbeScorer(model, device="cuda", precision="fp16", batch=1024, channels_last=True)
            prec = "fp16"
        t_run = time.time()
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for k, (ci, s, e) in enumerate(todo):
                if os.path.exists(os.path.join(chunk_dir, "STOP")):
                    stopped = "STOP file"
                    PX.log("STOP file: stopping before the next chunk", chunks_left=len(todo) - k)
                    break
                t0 = time.time()
                res, t_gpu = PX.run_chunk(pool, sc, games[s:e], 4096)
                secs = time.time() - t0
                tmp = path(ci) + ".tmp.npz"
                np.savez(tmp, format=np.array(P.CHUNK_FORMAT), set=np.array(set_name), sides=np.array(PX.SIDES),
                         ref=refs[s:e], precision=np.array(prec), meta_sha256=np.array(P.neutral_table_sha()),
                         ckpt_bytes=np.array(ck["bytes"]),
                         code_sha256=np.array(json.dumps(P.code_shas(), sort_keys=True)),
                         keys_sha256=np.array(ksha), mode=np.array(mode), created_utc=np.array(H.utc_now()),
                         seconds=np.array(secs), **res)
                os.replace(tmp, path(ci))
                el = time.time() - t_run
                PX.log("chunk done", chunk=ci, games=e - s, failed=int(res["failed"].sum()),
                       positions=int(len(res["pos_side"])), seconds=round(secs, 1), gpu_seconds=round(t_gpu, 1),
                       progress="%d/%d" % (k + 1, len(todo)),
                       eta_min=round(el / (k + 1) * (len(todo) - k - 1) / 60, 1))
        sc.close()
    if stopped:
        return 0
    # ---- 3 every chunk there: the report and the seam features
    recs = [F.chunk_record(path(ci)) for ci, _s, _e in chunks]
    X, usable, failed, have, cnt = F.load_side0([path(ci) for ci, _s, _e in chunks], len(games), set_name)
    n_seam = min(SEAM_GAMES, len(games))
    os.makedirs(feat_dir, exist_ok=True)
    seam = os.path.join(feat_dir, "val_seam_feats.npz")
    np.savez(seam, side0=X[:n_seam], counts0=cnt[:n_seam], failed=failed[:n_seam], ref=np.arange(n_seam),
             keys_sha256=np.array(ksha))
    rep = {"format": F.FORMATS["val_report"], "mode": mode, "complete": bool(have.all()), "games": len(games),
           "chunks": recs, "failed": int(failed.sum()), "no_target_moves": int(((cnt == 0) & ~failed).sum()),
           "usable": int(usable.sum()), "keys_sha256": ksha, "inputs_sha256": in_sha,
           "frozen_config_sha256": fc_sha, "set": set_name, "max_games": a.max_games or None,
           "seam": {"file": os.path.relpath(seam, REPO).replace(os.sep, "/"), "sha256": sha_file(seam),
                    "games": n_seam},
           "created_utc": H.utc_now(), "tool_sha256": sha_file(os.path.abspath(__file__))}
    with open(os.path.join(feat_dir, "val_report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
        f.write("\n")
    print(json.dumps({k: rep[k] for k in ("mode", "complete", "games", "failed", "no_target_moves", "usable")}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sets")
    x = sub.add_parser("extract")
    x.add_argument("--set", required=True)
    x.add_argument("--workers", type=int, default=12)
    x.add_argument("--max-games", type=int, default=0)
    x.add_argument("--smoke", action="store_true")
    x.add_argument("--candidates", default=None)
    x.add_argument("--questions", default=None)
    x.add_argument("--out-dir", default=None)
    x.add_argument("--fake-scorer", action="store_true")
    sub.add_parser("status")
    m = sub.add_parser("manifest")
    m.add_argument("--set", required=True)
    a = ap.parse_args()
    if a.cmd == "extract" and a.set != "val" and (a.smoke or a.candidates or a.questions or a.out_dir
                                                  or a.fake_scorer):
        raise SystemExit("--smoke / --candidates / --questions / --out-dir / --fake-scorer are for --set val only "
                         "(the mock extraction takes --workers and --max-games, as committed in 15b3e98)")
    return {"sets": cmd_sets, "extract": cmd_extract, "status": cmd_status, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
