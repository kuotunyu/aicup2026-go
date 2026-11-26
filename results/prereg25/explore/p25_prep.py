"""prereg25 (CPU, .venv): seam G of the GPU run and the V feature arrays (PLAN.md 2.3, 2.4). No model, no score.

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe
    S=results/prereg25/explore
    $S/p25_prep.py seam   -> $S/seam_g.json   (the 400 re-extracted P3 probe games vs the original P3 chunks and the
                                              probe_X.npy rows; passes when failed / counts / n_moves / refs are equal
                                              and the largest |difference| of the Q09-standardised features <= 0.05)
    $S/p25_prep.py prep   -> data/prereg25/prep/{v_X.npy, v_meta.npz, prep_record.json}

Reads: data/prereg25/seam, data/prereg09/p3/probe (chunks 0, 1), data/prereg17/prep (probe_X / probe_meta),
data/prereg25/v (chunks, sets.json), Q09's frozen p1e_map.npz. Never data/local.
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p25_lib as L  # noqa: E402  (tools/ on sys.path)
import numpy as np  # noqa: E402

import p09_forward as FW  # noqa: E402
import p09_map as M  # noqa: E402

REPO = L.REPO
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
Q09_CHECK = os.path.join(REPO, "results", "prereg09", "rank", "check_report.json")
PROBE_PREP = os.path.join(REPO, "data", "prereg17", "prep")
P3_CHUNKS = os.path.join(REPO, "data", "prereg09", "p3", "probe")
SEAM_DIR = os.path.join(REPO, "data", "prereg25", "seam", "probe")
V_DIR = os.path.join(REPO, "data", "prereg25", "v")
V_SETS = os.path.join(V_DIR, "sets.json")
PREP = os.path.join(REPO, "data", "prereg25", "prep")
SEAM_OUT = os.path.join(HERE, "seam_g.json")
SEAM_GAMES = 400
TOL_FEAT = 0.05          # PREREG_08 5.2 GPU seam (tools/t08_rank.TOL_FEAT): standardised features
CHUNK = 200


def log(*x):
    print(*x, flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def sha_file(path):
    h = hashlib.sha256()
    with open(L.guard(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_chunk(path):
    with np.load(L.guard(path), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in ("ref", "failed", "counts", "n_moves", "means", "pos_game", "pos_side",
                                               "pos_turn", "pos_logp", "pos_entropy", "precision", "batch",
                                               "torch")}


def cat(chunks, k):
    return np.concatenate([c[k] for c in chunks])


# ---------------------------------------------------------------- seam G (PLAN.md 2.3)
def cmd_seam(_a):
    t0 = time.time()
    n_ch = SEAM_GAMES // CHUNK
    new = [load_chunk(os.path.join(SEAM_DIR, f"chunk_{i:05d}.npz")) for i in range(n_ch)]
    old = [load_chunk(os.path.join(P3_CHUNKS, f"chunk_{i:05d}.npz")) for i in range(n_ch)]
    eq = {k: bool(np.array_equal(cat(new, k), cat(old, k))) for k in ("ref", "failed", "counts", "n_moves")}
    refs = cat(new, "ref")
    eq["refs_are_0_to_399"] = bool(np.array_equal(refs, np.arange(SEAM_GAMES)))
    bitwise = bool(np.array_equal(cat(new, "means"), cat(old, "means")))
    mp = FW.load_map(Q09_MAP)
    # probe_X rows of these games (09's prep: src p3_probe, ref)
    want = (json.load(open(Q09_CHECK, encoding="utf-8")).get("prep") or {}).get("files_sha256") or {}
    px_sha = sha_file(os.path.join(PROBE_PREP, "probe_X.npy"))
    with np.load(L.guard(os.path.join(PROBE_PREP, "probe_meta.npz")), allow_pickle=False) as z:
        src, pref = np.asarray(z["src"]).astype(str), np.asarray(z["ref"]).astype(np.int64)
    PX = np.load(L.guard(os.path.join(PROBE_PREP, "probe_X.npy")), mmap_mode="r")
    row_of = {int(r): i for i, (s, r) in enumerate(zip(src, pref)) if s == "p3_probe" and r < SEAM_GAMES}
    usable = FW.usable_games(cat(new, "failed"), cat(new, "counts"))
    Xn = FW.x_raw32(cat(new, "means"), cat(new, "counts"))
    Xo = FW.x_raw32(cat(old, "means"), cat(old, "counts"))
    games = [g for g in range(SEAM_GAMES) if usable[g]]
    in_probe = [g for g in games if g in row_of]
    Xp = np.asarray(PX[[row_of[g] for g in in_probe]], dtype=np.float32)
    stdn = FW.standardise_x(Xn[in_probe], mp["mu_x"], mp["sd_x"])
    stdp = FW.standardise_x(Xp, mp["mu_x"], mp["sd_x"])
    stdo = FW.standardise_x(Xo[in_probe], mp["mu_x"], mp["sd_x"])
    max_feat = float(np.max(np.abs(stdn - stdp)))
    old_vs_probe = float(np.max(np.abs(stdo - stdp)))
    G = SEAM_GAMES
    exn = np.concatenate([FW.ex_raw32({k: c[k] for k in ("pos_game", "pos_side", "pos_turn", "pos_logp",
                                                        "pos_entropy")}, len(c["ref"])) for c in new])
    exo = np.concatenate([FW.ex_raw32({k: c[k] for k in ("pos_game", "pos_side", "pos_turn", "pos_logp",
                                                        "pos_entropy")}, len(c["ref"])) for c in old])
    lpn = FW.map_logp(mp, Xn[games], exn[games])
    lpo = FW.map_logp(mp, Xo[games], exo[games])
    ok = bool(all(eq.values()) and len(in_probe) == len(games) and px_sha == want.get("probe_X.npy")
              and max_feat <= TOL_FEAT)
    rec = {"format": "c10-prereg25 explore seam G v1",
           "what": "PLAN.md 2.3 seam G: the first 400 P3 probe games re-extracted by p25_extract.py (p1_extract "
                   "unchanged) vs the original P3 chunks and the probe_X.npy rows",
           "games": G, "usable_games": len(games), "usable_games_in_probe_X": len(in_probe),
           "equal": eq, "probe_X_sha256_is_09": bool(px_sha == want.get("probe_X.npy")),
           "max_abs_std_feature_vs_probe_X": max_feat, "tol_std_feature": TOL_FEAT,
           "record_only": {"bitwise_equal_means_vs_original_chunks": bitwise,
                           "original_chunks_vs_probe_X_max_abs_std_feature": old_vs_probe,
                           "max_abs_lp_P1TO_E_q09_map": float(np.max(np.abs(lpn - lpo))),
                           "new_run": {"precision": str(new[0]["precision"]), "batch": int(new[0]["batch"]),
                                       "torch": str(new[0]["torch"])},
                           "original_run": {"precision": str(old[0]["precision"]), "batch": int(old[0]["batch"]),
                                            "torch": str(old[0]["torch"])}},
           "passed": ok, "seconds": round(time.time() - t0, 1)}
    write_json(SEAM_OUT, rec)
    log(json.dumps(rec, indent=1))
    return 0 if ok else 1


# ---------------------------------------------------------------- V prep (PLAN.md 2.4)
def cmd_prep(_a):
    t0 = time.time()
    if os.path.exists(os.path.join(PREP, "prep_record.json")):
        raise SystemExit("data/prereg25/prep/prep_record.json exists")
    with open(L.guard(V_SETS), encoding="utf-8") as f:
        sets = json.load(f)
    games = sets["probe"]["games"]
    n_ch = (len(games) + CHUNK - 1) // CHUNK
    files = [os.path.join(V_DIR, "probe", f"chunk_{i:05d}.npz") for i in range(n_ch)]
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} V chunks missing (first {missing[0]})")
    n = sum(M.usable_count(p, ()) for p in files)
    os.makedirs(PREP, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(PREP, "v_X.npy.tmp"), mode="w+", dtype=np.float32,
                                  shape=(n, FW.X_DIM))
    refs, ys, players, cols, shas = [], [], [], [], []
    i = 0
    unusable = 0
    for k, p in enumerate(files):
        c = M.read_chunk_rows(p, None, (), games)
        expect = np.arange(k * CHUNK, min(len(games), (k + 1) * CHUNK))
        if not np.all(np.isin(c["ref"], expect)):
            raise SystemExit(f"{p}: refs outside the chunk's games")
        g = len(c["ref"])
        X[i:i + g] = c["X"]
        refs.append(c["ref"])
        ys.append(c["y"])
        players.append(c["players"])
        cols.append(np.array([games[j]["color"] for j in c["ref"]]))
        shas.append(c["sha"])
        unusable += c["unusable"]
        i += g
    if i != n:
        raise SystemExit(f"wrote {i} rows, expected {n}")
    X.flush()
    del X
    os.replace(os.path.join(PREP, "v_X.npy.tmp"), os.path.join(PREP, "v_X.npy"))
    np.savez(os.path.join(PREP, "v_meta.npz"), ref=np.concatenate(refs), y=np.concatenate(ys),
             players=np.concatenate(players), color=np.concatenate(cols))
    rec = {"format": "c10-prereg25 explore v prep v1",
           "what": "PLAN.md 2.4: V chunks -> raw TO features (tools/p09_map.read_chunk_rows, unchanged)",
           "sets_sha256": sha_file(V_SETS), "games_sha256": sets["probe"]["games_sha256"], "chunks": n_ch,
           "games": len(games), "rows": int(n), "unusable": int(unusable), "chunk_sha256": shas,
           "files_sha256": {f: sha_file(os.path.join(PREP, f)) for f in ("v_X.npy", "v_meta.npz")},
           "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(PREP, "prep_record.json"), rec)
    log(json.dumps({k: rec[k] for k in ("chunks", "games", "rows", "unusable", "files_sha256", "seconds")}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seam")
    sub.add_parser("prep")
    a = ap.parse_args(argv)
    return {"seam": cmd_seam, "prep": cmd_prep}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
