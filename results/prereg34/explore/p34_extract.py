"""prereg34 (GPU): results/prereg08/p1/p1_extract.py, unchanged, with OGS games as one extra source (PLAN.md 3).

    Git Bash, repository root (subst drive Q:); PYTHONDONTWRITEBYTECODE=1
    GPU python .venv-gpu/Scripts/python.exe, CPU python .venv/Scripts/python.exe;  S=results/prereg34/explore
    CPU  $S/p34_extract.py sets --games <sample_games.csv.gz under the repo> [--out data/prereg34/p1/sets.json]
    GPU  $S/p34_extract.py run --set probe --sets <sets.json> --out-dir <dir> --workers 12 [...]   (= p1_extract run)
    CPU  $S/p34_extract.py seam [--new-dir data/prereg34/seam/probe] [--where 4090|colab]   -> $S/seam_g34.json
    CPU  $S/p34_extract.py colab-bundle [--sets data/prereg34/p1/sets.json]   -> data/prereg34/colab/upload/...
    CPU  $S/p34_extract.py seam-colab --new-dir <dir>/probe --ref <seam_ref.npz> --out <json>   (the notebook's check)

run: every argument goes to p1_extract.main. The only change: p1_extract.set_games is wrapped so that a probe whose
source is 'ogs' (or 'p3_seam', the Colab copy of the seam games) takes its games from the games file named in sets.json
(a gzip CSV; player and colour checked against sets.json; ref = index in the probe list). Any other source goes to the
original function unchanged, so seam G34 re-runs the P3 probe (source 'fit') through this same entry point.

seam: the prereg25 seam G comparison (p25_prep.cmd_seam, imported, unchanged) on the re-extracted P3 probe chunks 0-1
vs probe_X.npy; passes when failed / counts / n_moves / refs are equal and the largest |difference| of the
Q09-standardised features <= 0.05. Only the input folder and the output name differ.

seam-colab: the same comparison against the original P3 chunks 0-1 (seam G of prereg25 recorded them equal to the
probe_X rows, max difference 0.0), standard library + numpy only, so it runs on Colab before the OGS games.

Never reads data/local; never runs tools/score.py.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import time
import zipfile

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg08", "p1")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402

import p1_extract as E  # noqa: E402  (p1_lib pin check; hs_lib's data/local guard)

FILE_SOURCES = ("ogs", "p3_seam")
SETS_FORMAT = E.SETS_FORMAT                     # "c10-prereg08 p1 sets v1" (p1_extract.load_sets accepts only this)
OGS_COLUMNS = ("player_id", "game_id", "ogs_rank", "color", "start_time", "speed", "n_moves", "sgf")   # p34_ogs
SEAM_COLUMNS = ("player_id", "game_id", "rank", "color", "sgf_content")                                # c10data
SEAM_GAMES = 400
CHUNK = E.CHUNK
TOL_FEAT = 0.05            # PREREG_08 5.2 GPU seam (tools/t08_rank.TOL_FEAT), as prereg25 seam G
P1_DIR = os.path.join(REPO, "data", "prereg34", "p1")
SETS_OUT = os.path.join(P1_DIR, "sets.json")
SEAM_DIR = os.path.join(REPO, "data", "prereg34", "seam", "probe")
SEAM_OUT = os.path.join(HERE, "seam_g34.json")
P3_SETS = os.path.join(REPO, "data", "prereg09", "p3", "sets.json")
P3_CHUNKS = os.path.join(REPO, "data", "prereg09", "p3", "probe")
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
KIT_RECORD = os.path.join(REPO, "results", "test11", "kit_manifest.json")
BUNDLE_DIR = os.path.join(REPO, "data", "prereg34", "colab")
BUNDLE_NAME = "C10_p34_bundle.zip"
BUNDLE_RECORD = os.path.join(HERE, "colab_bundle.json")
BUNDLE_FORMAT = "c10-prereg34 colab bundle v1"
KIT_ROOT = "c10kit"                              # the g11 kit's zip root (tools/g11_gpu_fallback.py KIT_ROOT)
FORBIDDEN = ("/data/local", "rank_val_answers", "player_val_answers")

_ORIGINAL = E.set_games


def guard(p):
    q = os.path.abspath(p).replace("\\", "/").lower()
    if any(f in q for f in FORBIDDEN):
        raise SystemExit(f"refused path {p}")
    return p


def sha_raw(path):
    h = hashlib.sha256()
    with open(guard(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def repo_path(rel_or_abs):
    return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(REPO, *rel_or_abs.split("/"))


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_games_file(path):
    """gzip CSV rows (dicts). The OGS sample (p34_ogs COLUMNS + bin, player_ogs_median) or the seam copy (c10data
    COLUMNS)."""
    csv.field_size_limit(1 << 26)
    with gzip.open(guard(path), "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- the wrapper (PLAN.md 3)
def set_games(sets, name):
    s = sets[name]
    if s.get("source") not in FILE_SOURCES:
        return _ORIGINAL(sets, name)
    path = repo_path(s["games_file"])
    if sha_raw(path) != s["games_file_sha256"]:
        raise E.P.P1Error(f"{s['games_file']}: SHA-256 differs from sets.json")
    sgf_col = "sgf" if s["source"] == "ogs" else "sgf_content"
    rows = {}
    for r in read_games_file(path):
        k = (r["game_id"], r["color"])
        if k in rows:
            raise E.P.P1Error(f"{s['games_file']}: game {k} listed twice")
        rows[k] = r
    out = []
    for j, g in enumerate(s["games"]):
        r = rows.get((g["game_id"], g["color"]))
        if r is None:
            raise E.P.P1Error(f"{name} game {j}: {g['game_id']} colour {g['color']} not in {s['games_file']}")
        if r["player_id"] != g["player_id"] or r["color"] != g["color"] \
                or ("rank" in g and r.get("rank") != g["rank"]):
            raise E.P.P1Error(f"{name} game {j}: player or colour differs from sets.json")
        out.append({"player_id": r["player_id"], "game_id": r["game_id"], "rank": r.get("rank", "ogs"),
                    "color": r["color"], "sgf_content": r[sgf_col], "ref": j})
    return out


E.set_games = set_games


# ---------------------------------------------------------------- sets.json of the OGS games
def build_ogs_sets(games_path, games_rel):
    rows = read_games_file(games_path)
    if not rows or tuple(rows[0].keys())[:len(OGS_COLUMNS)] != OGS_COLUMNS:
        raise SystemExit(f"{games_rel}: not a p34_ogs sample file (columns {tuple(rows[0].keys()) if rows else ()})")
    games = [{"game_id": r["game_id"], "player_id": r["player_id"], "color": r["color"],
              "ogs_rank": r["ogs_rank"]} for r in rows]
    if len({(g["game_id"], g["color"]) for g in games}) != len(games):
        raise SystemExit(f"{games_rel}: a (game, colour) is listed twice")
    if any(g["color"] not in ("B", "W") or not g["player_id"].startswith("ogs_")
           or not g["game_id"].startswith("ogs_") for g in games):
        raise SystemExit(f"{games_rel}: unexpected player / game id or colour")
    return {"format": SETS_FORMAT, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "probe": {"source": "ogs", "games_file": games_rel, "games_file_sha256": sha_raw(games_path),
                      "games": games,
                      "what": "prereg34 OGS sample (p34_ogs.py sample), file order; ref = index; no official label"},
            "counts": {"probe": len(games), "probe_players": len({g["player_id"] for g in games}), "bank": 0,
                       "mono": 0}}


def rel_to_repo(path):
    try:
        rel = os.path.relpath(os.path.abspath(path), REPO).replace("\\", "/")
    except ValueError:                           # another drive
        rel = ".."
    if rel.startswith(".."):
        raise SystemExit(f"{path} is not inside the repository (the Colab bundle mirrors repository paths)")
    return rel


def cmd_sets(a):
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; not overwritten")
    s = build_ogs_sets(a.games, rel_to_repo(a.games))
    write_json(out, s)
    print(json.dumps({"written": out, "sha256": sha_raw(out), "counts": s["counts"]}), flush=True)
    return 0


# ---------------------------------------------------------------- seam G34
def cmd_seam(a):
    p25 = os.path.join(REPO, "results", "prereg25", "explore")
    if p25 not in sys.path:
        sys.path.insert(0, p25)
    import p25_prep as P25                       # local only (tools/, data/prereg17/prep, Q09 map)
    tmp = os.path.join(REPO, "data", "prereg34", "seam_g34_raw.json")
    P25.SEAM_DIR, P25.SEAM_OUT = os.path.abspath(a.new_dir), tmp
    rc = P25.cmd_seam(a)
    with open(tmp, encoding="utf-8") as f:
        rec = json.load(f)
    rec = {"format": "c10-prereg34 explore seam G34 v1",
           "what": "PLAN.md 3 seam G34: the first 400 P3 probe games re-extracted by p34_extract.py (p1_extract "
                   "unchanged) vs the original P3 chunks and the probe_X.npy rows; comparison = "
                   "results/prereg25/explore/p25_prep.py cmd_seam (imported, unchanged)",
           "where": a.where, "new_dir": rel_to_repo(a.new_dir), **{k: v for k, v in rec.items()
                                                                   if k not in ("format", "what")}}
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; not overwritten")
    write_json(a.out, rec)
    os.remove(tmp)
    return rc


def load_chunk(path):
    with np.load(guard(path), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in ("ref", "failed", "counts", "n_moves", "means")}


def std_x(means, counts, mu, sd):
    """tools/p09_forward x_raw32 then standardise_x: [T | O] float32 -> float64, O = NaN without an opponent move,
    (X - mu) / sd, non-finite -> 0."""
    means = np.asarray(means)
    T = means[:, 0, :].astype(np.float64)
    O = means[:, 1, :].astype(np.float64)
    O[np.asarray(counts)[:, 1] == 0] = np.nan
    X = np.concatenate([T, O], axis=1).astype(np.float32).astype(np.float64)
    Xs = (X - mu) / sd
    Xs[~np.isfinite(Xs)] = 0.0
    return Xs


def seam_compare(new, ref):
    """new: list of chunk dicts (chunks 0, 1); ref: dict of the reference arrays. -> record."""
    cat = {k: np.concatenate([c[k] for c in new]) for k in ("ref", "failed", "counts", "n_moves", "means")}
    eq = {k: bool(cat[k].shape == ref[k].shape and np.array_equal(cat[k], ref[k]))
          for k in ("ref", "failed", "counts", "n_moves")}
    eq["refs_are_0_to_399"] = bool(np.array_equal(cat["ref"], np.arange(SEAM_GAMES)))
    if not all(eq.values()):
        return {"equal": eq, "max_abs_std_feature": None, "passed": False}
    use = (~cat["failed"].astype(bool)) & (cat["counts"][:, 0] > 0)
    d = np.abs(std_x(cat["means"][use], cat["counts"][use], ref["mu_x"], ref["sd_x"])
               - std_x(ref["means"][use], ref["counts"][use], ref["mu_x"], ref["sd_x"]))
    mx = float(d.max()) if d.size else 0.0
    return {"equal": eq, "usable_games": int(use.sum()), "max_abs_std_feature": mx, "tol_std_feature": TOL_FEAT,
            "bitwise_equal_means": bool(np.array_equal(cat["means"], ref["means"])),
            "passed": bool(mx <= TOL_FEAT and use.sum() > 0)}


def cmd_seam_colab(a):
    new = [load_chunk(os.path.join(a.new_dir, f"chunk_{i:05d}.npz")) for i in range(SEAM_GAMES // CHUNK)]
    with np.load(a.ref, allow_pickle=False) as z:
        ref = {k: np.asarray(z[k]) for k in z.files}
    with np.load(os.path.join(a.new_dir, "chunk_00000.npz"), allow_pickle=False) as z:
        run = {k: str(z[k]) for k in ("precision", "batch", "torch", "device")}
    rec = {"format": "c10-prereg34 colab seam check v1",
           "what": "the 400 seam games on this GPU vs the original P3 chunks 0-1 (4090), Q09-standardised features",
           **seam_compare(new, ref), "new_run": run}
    write_json(a.out, rec)
    print(json.dumps(rec, indent=1), flush=True)
    return 0 if rec["passed"] else 1


# ---------------------------------------------------------------- the Colab bundle
def kit_record():
    with open(KIT_RECORD, encoding="utf-8") as f:
        return json.load(f)


def seam_files(work):
    """seam_games.csv.gz, seam_sets.json, seam_ref.npz in `work` (repo-relative data/prereg34/p1 in the bundle)."""
    p3 = E.load_sets(P3_SETS)
    orig = _ORIGINAL(p3, "probe")[:SEAM_GAMES]
    gpath = os.path.join(work, "seam_games.csv.gz")
    with open(gpath, "wb") as fo_raw:
        fo = io.TextIOWrapper(gzip.GzipFile(fileobj=fo_raw, mode="wb", compresslevel=6, mtime=0),
                              encoding="utf-8", newline="")
        wr = csv.writer(fo, lineterminator="\n")
        wr.writerow(SEAM_COLUMNS)
        for r in orig:
            wr.writerow([r[c] for c in SEAM_COLUMNS])
        fo.close()
    seam_sets = {"format": SETS_FORMAT, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "probe": {"source": "p3_seam", "games_file": "data/prereg34/p1/seam_games.csv.gz",
                           "games_file_sha256": sha_raw(gpath), "games": p3["probe"]["games"][:SEAM_GAMES],
                           "what": "the first 400 P3 probe games (data/prereg09/p3/sets.json, source fit), copied "
                                   "for Colab (no training CSV there)"},
                 "counts": {"probe": SEAM_GAMES, "bank": 0, "mono": 0}}
    spath = os.path.join(work, "seam_sets.json")
    write_json(spath, seam_sets)
    # the copy must give exactly the original rows through the wrapper
    seam_sets["probe"]["games_file"] = gpath
    got = set_games(seam_sets, "probe")
    for a_, b_ in zip(got, orig):
        if any(a_[k] != b_[k] for k in ("player_id", "game_id", "rank", "color", "sgf_content", "ref")):
            raise SystemExit("seam copy differs from p1_extract.set_games on the P3 probe")
    if len(got) != SEAM_GAMES:
        raise SystemExit("seam copy: wrong game count")
    old = [load_chunk(os.path.join(P3_CHUNKS, f"chunk_{i:05d}.npz")) for i in range(SEAM_GAMES // CHUNK)]
    with np.load(guard(Q09_MAP), allow_pickle=False) as z:
        mu, sd = np.asarray(z["mu_x"]), np.asarray(z["sd_x"])
    rpath = os.path.join(work, "seam_ref.npz")
    np.savez(rpath, **{k: np.concatenate([c[k] for c in old]) for k in ("ref", "failed", "counts", "n_moves",
                                                                         "means")}, mu_x=mu, sd_x=sd)
    return {"seam_games.csv.gz": gpath, "seam_sets.json": spath, "seam_ref.npz": rpath}


def cmd_colab_bundle(a):
    t0 = time.time()
    kit = kit_record()
    want_code = kit["manifest"]["constants"]["p1_code_record"]
    if E.P.code_shas() != want_code:
        raise SystemExit("the repository's P1 code differs from the g11 kit's (C10_gpu_kit_v1.zip): rebuild the kit")
    sets = E.load_sets(a.sets)
    s = sets["probe"]
    if s.get("source") != "ogs":
        raise SystemExit(f"{a.sets}: probe source {s.get('source')!r}, not 'ogs'")
    gpath = repo_path(s["games_file"])
    if sha_raw(gpath) != s["games_file_sha256"]:
        raise SystemExit(f"{s['games_file']}: SHA-256 differs from sets.json")
    work = os.path.join(BUNDLE_DIR, "work")
    os.makedirs(work, exist_ok=True)
    seam = seam_files(work)
    files = {"results/prereg34/explore/p34_extract.py": os.path.abspath(__file__),
             "data/prereg34/p1/sets.json": os.path.abspath(a.sets),
             s["games_file"]: gpath,
             **{f"data/prereg34/p1/{k}": v for k, v in seam.items()}}
    man = {"format": BUNDLE_FORMAT, "kit_id": kit["kit_id"], "kit_zip_sha256": kit["zip"]["sha256"],
           "p1_code_record": want_code, "ogs_games": len(s["games"]), "seam_games": SEAM_GAMES,
           "files": {rel: sha_raw(src) for rel, src in sorted(files.items())}}
    mbytes = (json.dumps(man, indent=1, sort_keys=True) + "\n").encode("utf-8")
    up = os.path.join(BUNDLE_DIR, "upload")
    os.makedirs(up, exist_ok=True)
    zpath = os.path.join(up, BUNDLE_NAME)
    with zipfile.ZipFile(zpath + ".tmp", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{KIT_ROOT}/P34_BUNDLE_MANIFEST.json", mbytes)
        for rel, src in sorted(files.items()):
            zf.write(src, f"{KIT_ROOT}/{rel}")
    os.replace(zpath + ".tmp", zpath)
    rec = {"format": BUNDLE_FORMAT, "zip": {"file": BUNDLE_NAME, "sha256": sha_raw(zpath),
                                            "bytes": os.path.getsize(zpath)},
           "manifest": man, "seconds": round(time.time() - t0, 1)}
    write_json(BUNDLE_RECORD, rec)
    print(json.dumps({k: rec[k] for k in ("zip", "seconds")}), flush=True)
    return 0


# ---------------------------------------------------------------- entry
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("sets", "seam", "seam-colab", "colab-bundle"):
        return E.main(argv)                      # run / status / check-smoke / p1_extract's own sets
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--games", required=True)
    s.add_argument("--out", default=SETS_OUT)
    m = sub.add_parser("seam")
    m.add_argument("--new-dir", default=SEAM_DIR)
    m.add_argument("--where", default="4090", choices=["4090", "colab"])
    m.add_argument("--out", default=SEAM_OUT)
    c = sub.add_parser("seam-colab")
    c.add_argument("--new-dir", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--out", required=True)
    b = sub.add_parser("colab-bundle")
    b.add_argument("--sets", default=SETS_OUT)
    a = ap.parse_args(argv)
    return {"sets": cmd_sets, "seam": cmd_seam, "seam-colab": cmd_seam_colab, "colab-bundle": cmd_colab_bundle}[
        a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
