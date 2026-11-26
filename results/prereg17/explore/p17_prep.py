"""prereg17 exploration (competition 2): the TabICL inputs of PLAN.md 2.1 (CPU, .venv). No label of an inner game is
read here and no candidate number is computed.

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg17/explore
    $S/p17_prep.py prep            -> data/prereg17/prep/ (09's prep arrays rebuilt; seam P) and
                                      data/prereg17/feat/ (PCA 128 on the 270,000 probe rows, context 30,000, inner Z)
    $S/p17_prep.py prep --smoke    -> data/prereg17/smoke/feat/ (the same files with Z replaced by default_rng(12345)
                                      normal noise and a 1,000-game context; pipeline only, never a result)

Reads: the P1 / P2 / P3 probe chunks and the bank / mono chunks (SHA-256 against 08's inner_manifest.json and 09's
p3_manifest.json while reading, tools/p09_map.build_prep), Q09's frozen p1e_map.npz (mu / sd), data/prereg12/q09_inner/
inner_X.npy (byte comparison only). Never data/local.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
for p in (TOOLS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p09_rank_fuse as r  # noqa: E402
import p09_forward as FW  # noqa: E402
import p09_map as M  # noqa: E402
import p17_lib as K  # noqa: E402

R = r.R
DATA = os.path.join(REPO, "data", "prereg17")
PREP = os.path.join(DATA, "prep")
FEAT = os.path.join(DATA, "feat")
SMOKE_FEAT = os.path.join(DATA, "smoke", "feat")
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
Q09_CHECK = os.path.join(REPO, "results", "prereg09", "rank", "check_report.json")
Q09_FROZEN = os.path.join(REPO, "results", "prereg09", "rank", "frozen_config.json")
Q12_INNER_X = os.path.join(REPO, "data", "prereg12", "q09_inner", "inner_X.npy")
BLOCK = 8192
FORMAT = "c10-prereg17 explore feat v1"
FORBIDDEN = ("data/local", "rank_val_answers", "player_val_answers")


def guard(p):
    q = os.path.abspath(p).replace("\\", "/").lower()
    for f in FORBIDDEN:
        if ("/" + f) in q:
            raise SystemExit(f"refused path {p}")
    return p


def sha_file(path):
    h = hashlib.sha256()
    with open(guard(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def git_head():
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- 1 rebuild 09's prep (seam P)
def rebuild_prep(log):
    """tools/p09_map.py cmd_prep's assembly (P1 + P2 + P3 as the committed manifest decided), written by the
    unchanged build_prep into data/prereg17/prep/."""
    rec_path = os.path.join(PREP, "prep_record.json")
    if os.path.exists(rec_path):
        log(f"prep exists: {rec_path}")
        return r.read_json(rec_path)
    man = r.read_json(r.MANIFEST)
    sets = r.probe_sets_from_manifest(man)
    if sets != list(r.PROBE_SETS):
        raise SystemExit(f"09's manifest uses {sets}, the frozen map used P1+P2+P3")
    man08 = R.read_json(r.P08_MANIFEST)
    games = {}
    for s, path in (("p1_probe", R.P1_SETS), ("p2_probe", R.P2_SETS), ("p3_probe", r.P3_SETS)):
        with open(guard(path), encoding="utf-8") as f:
            games[s] = json.load(f)["probe"]["games"]
    dup08 = man08.get("duplicates") or {}
    parts = []
    for s in sets:
        if s == "p3_probe":
            fl = [(guard(r.path_of(x["file"])), x["sha256"]) for x in man["p3_chunks"]["chunks"]]
            dups = (man.get("duplicates") or {}).get("p3_probe") or []
        else:
            fl = [(guard(r.path_of(x["file"])), x["sha256"]) for x in man08["sets"][s]["chunks"]]
            dups = dup08.get(s) or []
        parts.append((s, fl, games[s], dups))
    inner = [(s, [(guard(r.path_of(x["file"])), x["sha256"]) for x in man08["sets"][s]["chunks"]])
             for s in ("bank", "mono")]
    return M.build_prep(PREP, parts, inner, {"what": "prereg17: 09's prep rebuilt (PLAN.md 2.1 item 1)",
                                             "probe_used": "P1+P2+P3", "probe_sets": sets}, log=log)


def seam_p():
    want = (r.read_json(Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = {f: sha_file(os.path.join(PREP, f)) for f in r.PREP_FILES}
    q12 = sha_file(Q12_INNER_X)
    ok = bool(want) and all(want.get(f) == got[f] for f in r.PREP_FILES) and q12 == got["inner_X.npy"]
    return {"ok": bool(ok), "want": want, "got": got, "q12_inner_X_sha256": q12}


# ---------------------------------------------------------------- 2 features
def std_blocks(X, rows, mu, sd):
    for s in range(0, len(rows), BLOCK):
        yield FW.standardise_x(np.asarray(X[rows[s:s + BLOCK]], dtype=np.float64), mu, sd)


def project_rows(X, rows, mu, sd, pmu, V):
    out = np.zeros((len(rows), V.shape[1]), dtype=np.float32)
    for i, Xs in enumerate(std_blocks(X, rows, mu, sd)):
        out[i * BLOCK:i * BLOCK + len(Xs)] = K.pca_project(Xs, pmu, V).astype(np.float32)
    return out


def save_npy(d, name, arr):
    p = os.path.join(d, name)
    np.save(p, arr)
    return sha_file(p)


def cmd_prep(a):
    t0 = time.time()
    r.blas_env_check(False)
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    mp = FW.load_map(Q09_MAP)
    map_sha = sha_file(Q09_MAP)
    if map_sha != r.read_json(Q09_FROZEN)["map"]["sha256"]:
        raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    prec = rebuild_prep(log)
    sp = seam_p()
    log(f"seam P {'ok' if sp['ok'] else 'FAILED'} ({time.time() - t0:.0f}s)")
    if not sp["ok"]:
        write_json(os.path.join(SMOKE_FEAT if a.smoke else FEAT, "feat_record.json"),
                   {"format": FORMAT, "status": "stop: seam P failed", "seam_p": sp})
        return 1
    prep = r.Prep(PREP)
    n_probe = len(prep.y)
    probe_rows = np.arange(n_probe)
    if set(np.unique(prep.src)) != set(r.PROBE_SETS):
        raise SystemExit("probe rows of another set")
    mu, sd = mp["mu_x"], mp["sd_x"]
    pca_path = os.path.join(FEAT, "pca.npz")
    if os.path.exists(pca_path):
        with np.load(pca_path, allow_pickle=False) as z:
            pmu, V, w, total, n_fit = z["mu"], z["V"], z["eigvals"], float(z["total_var"]), int(z["rows"])
        if V.shape != (FW.X_DIM, K.PCA_DIM) or pmu.shape != (FW.X_DIM,) or not np.all(np.isfinite(pmu)) \
                or not np.all(np.isfinite(V)):
            raise SystemExit(f"{pca_path}: wrong shape or non-finite values")
        log("PCA loaded")
    else:
        n_fit, pmu, C = K.pca_moments(std_blocks(prep.probe_X, probe_rows, mu, sd))
        V, w, total = K.pca_from_cov(C, K.PCA_DIM)
        os.makedirs(FEAT, exist_ok=True)
        tmp = pca_path + ".tmp.npz"
        np.savez(tmp, mu=pmu, V=V, eigvals=w, total_var=np.array(total), rows=np.array(n_fit))
        os.replace(tmp, pca_path)
        log(f"PCA fitted on {n_fit} probe rows ({time.time() - t0:.0f}s)")
    if n_fit != n_probe:
        raise SystemExit("PCA rows differ from the probe rows")
    ctx = K.stratified_context(prep.y, K.CONTEXT_PER_LEVEL, K.CONTEXT_SEED)
    inner_rows = np.arange(len(prep.inner_ref))
    out_dir = SMOKE_FEAT if a.smoke else FEAT
    os.makedirs(out_dir, exist_ok=True)
    if a.smoke:
        ctx = K.smoke_context(ctx, prep.y)
        rng = np.random.default_rng(K.SMOKE_SEED)
        Zc = rng.normal(size=(len(ctx), K.PCA_DIM)).astype(np.float32)
        Zi = rng.normal(size=(len(inner_rows), K.PCA_DIM)).astype(np.float32)
    else:
        Zc = project_rows(prep.probe_X, ctx, mu, sd, pmu, V)
        Zi = project_rows(prep.inner_X, inner_rows, mu, sd, pmu, V)
    yc = prep.y[ctx].astype(np.int64)
    files = {"Z_ctx.npy": save_npy(out_dir, "Z_ctx.npy", Zc), "y_ctx.npy": save_npy(out_dir, "y_ctx.npy", yc),
             "Z_inner.npy": save_npy(out_dir, "Z_inner.npy", Zi),
             "ctx_rows.npy": save_npy(out_dir, "ctx_rows.npy", ctx.astype(np.int64)),
             "ctx_players.npy": save_npy(out_dir, "ctx_players.npy", prep.players[ctx].astype(str)),
             "inner_ref.npy": save_npy(out_dir, "inner_ref.npy", prep.inner_ref.astype(np.int64))}
    rec = {"format": FORMAT, "created": r.now_iso(), "smoke": bool(a.smoke), "status": "ok",
           "smoke_note": "Z replaced by default_rng(12345) normal noise, context = first 100 per level of the registered"
                         " context: pipeline only" if a.smoke else None,
           "head": git_head(), "seconds": round(time.time() - t0, 1), "seam_p": sp,
           "prep_record_seconds": (prec or {}).get("seconds"), "q09_map_sha256": map_sha,
           "pca": {"dim": K.PCA_DIM, "rows": int(n_fit), "sources": sorted(set(prep.src.tolist())),
                   "explained_ratio": float(np.sum(w) / total), "eigvals_head": [float(x) for x in w[:10]],
                   "pca_sha256": sha_file(pca_path)},
           "context": {"rows": int(len(ctx)), "per_level": [int((yc == k).sum()) for k in range(K.L)],
                       "seed": K.CONTEXT_SEED, "players": int(len(np.unique(prep.players[ctx]))),
                       "sources": {s: int((prep.src[ctx] == s).sum()) for s in r.PROBE_SETS}},
           "inner_rows": int(len(inner_rows)), "files_sha256": files}
    write_json(os.path.join(out_dir, "feat_record.json"), rec)
    log(json.dumps({k: rec[k] for k in ("pca", "context", "inner_rows", "seconds")}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    return {"prep": cmd_prep}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
