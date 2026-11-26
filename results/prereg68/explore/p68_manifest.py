"""prereg68 (NOTES.md 9, audit data/advice/codex_1122b.md item 2): per-game provenance of the X2 / X3 vectors.

No torch, no score. Used by p68_embed.py (writes, at embedding time) and p68_eval.py (checks, at check-member /
judge / apply time).

Manifest of an official pool = one row per game, in the order the frozen embedder consumes them
(p36_embed.pool_games -> p35_apply.read_pool -> p36_lib.pool_items: candidate rows in file order, then every
question's games in question order):

    [pool, kind ("cand" | "q"), id (player_id | question_id), index (0-based game number within that candidate's
     rows / within the question), colour ("B" | "W"), sha256 of the raw SGF text (utf-8), target moves]

target moves = the target colour's non-pass moves (p36_lib.moves_of; prereg60's early=True samples every one of
them, p60_lib.target_moves60), i.e. the expected per-game `count` of a raw tta1 prereg60 vector file.

Sidecar <name>.provenance.json (written by p68_embed.py right after the frozen embedder replaced <name>.npz):
{format, arm, name, source, pool, src_csv_sha256, manifest (rows), manifest_sha256, npz_sha256, ckpt_path,
 ckpt_sha256 (the checkpoint file hashed again by the wrapper), ckpt_step, games, count_check, created_at}.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (os.path.join(REPO, "results", "prereg36", "explore"), os.path.join(REPO, "results", "prereg35", "explore")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SIDE_FORMAT = "c10-prereg68 provenance v2"   # v2: cold_start required (audit 1122e)
POOL_PREFIX = {"public": "pub2_", "private": "priv2_"}


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest_sha(rows):
    return sha_text(json.dumps(rows, separators=(",", ":"), ensure_ascii=True))


def target_moves(sgf, colour):
    import p36_lib as L36
    codes = np.asarray(L36.moves_of(sgf))
    white = codes >= 361
    return int((white if colour.upper() == "W" else ~white).sum())


def manifest_rows(pool, cands, qs, items=None):
    """Rows (see module doc) from the CSV dicts. If `items` (the list the embedder consumed) is given, the rows'
    (sgf, colour) sequence must equal it element for element (else ValueError)."""
    rows, seq, per = [], [], {}
    for r in cands:
        pid = r["player_id"]
        i = per.get(pid, 0)
        per[pid] = i + 1
        rows.append([pool, "cand", pid, i, r["color"].upper(), sha_text(r["sgf_content"]),
                     target_moves(r["sgf_content"], r["color"])])
        seq.append((r["sgf_content"], r["color"]))
    for q in qs:
        for i in range(1, int(q["num_games"]) + 1):
            s, c = q["sgf_%d" % i], q["color_%d" % i]
            rows.append([pool, "q", q["question_id"], i - 1, c.upper(), sha_text(s), target_moves(s, c)])
            seq.append((s, c))
    if items is not None:
        if len(items) != len(seq) or any(a[0] != b[0] or a[1] != b[1] for a, b in zip(items, seq)):
            raise ValueError("the embedder's item list differs from the CSV rows in order or content")
    return rows


def csv_shas(src, pool):
    return {n: sha_file(os.path.join(src, n)) for n in ("player_identification_candidates2_%s.csv" % pool,
                                                         "player_identification_test2_%s.csv" % pool)}


def rebuild(src, pool):
    """Manifest rebuilt from the official CSVs with the embedder's own readers (p35_apply.read_pool,
    p36_lib.pool_items)."""
    import p35_apply as pa
    import p36_lib as L36
    cands, qs = pa.read_pool(src, pool)
    items = L36.pool_items(cands, qs)[0]
    return manifest_rows(pool, cands, qs, items)


def sidecar_path(npz):
    return os.path.splitext(npz)[0] + ".provenance.json"


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def npz_meta(npz):
    with np.load(npz, allow_pickle=False) as z:
        return {"ckpt_sha256": str(z["ckpt_sha256"]), "ckpt_step": int(z["ckpt_step"]),
                "count": np.asarray(z["count"]).copy(), "n": int(len(z["count"]))}


def count_check(count, rows):
    want = np.array([r[6] for r in rows], np.int64)
    raw = np.asarray(count)
    if not np.issubdtype(raw.dtype, np.integer):
        return {"ok": False, "why": "count dtype %s is not integer" % raw.dtype}
    got = raw.astype(np.int64)
    if got.shape != want.shape:
        return {"ok": False, "why": "count shape %s, manifest %s" % (got.shape, want.shape)}
    bad = int((got != want).sum())
    return {"ok": bad == 0 and bool((got > 0).all()), "mismatched_games": bad,
            "zero_count_games": int((got == 0).sum())}


def verify_pool(npz, src, pool, want_ckpt_sha, rows=None):
    """Problems (list of str) of an official-pool vector file against its sidecar and the CSVs of `src`.
    rows: a manifest already rebuilt from the CSVs (else rebuilt here)."""
    side = sidecar_path(npz)
    if not os.path.exists(npz):
        return ["%s missing" % npz]
    if not os.path.exists(side):
        return ["%s: no provenance sidecar (made outside p68_embed.py?)" % os.path.basename(npz)]
    with open(side, encoding="utf-8") as f:
        s = json.load(f)
    p = []
    if s.get("format") != SIDE_FORMAT or s.get("source") != "pool" or s.get("pool") != pool:
        p.append("sidecar format / source / pool %r %r %r" % (s.get("format"), s.get("source"), s.get("pool")))
    if s.get("cold_start") is not True:
        p.append("provenance has no cold_start true (%r): not a verified from-scratch p68_embed.py run; delete and "
                 "embed again" % s.get("cold_start"))
    if s.get("manifest_sha256") != manifest_sha(s.get("manifest", [])):
        p.append("sidecar manifest does not hash to its manifest_sha256")
    if s.get("npz_sha256") != sha_file(npz):
        p.append("vector file sha256 differs from the sidecar")
    rows = rebuild(src, pool) if rows is None else rows
    if manifest_sha(rows) != s.get("manifest_sha256"):
        p.append("manifest rebuilt from %s/%s CSVs differs from the one bound at embedding" % (src, pool))
    if s.get("src_csv_sha256") != csv_shas(src, pool):
        p.append("official CSV sha256 differ from the ones read at embedding")
    m = npz_meta(npz)
    if m["ckpt_sha256"] != want_ckpt_sha or s.get("ckpt_sha256") != want_ckpt_sha:
        p.append("checkpoint sha256 of the file / sidecar is not the arm's")
    if m["n"] != len(rows):
        p.append("%d vectors for %d manifest games" % (m["n"], len(rows)))
    else:
        cc = count_check(m["count"], rows)
        if not cc["ok"]:
            p.append("per-game sample counts differ from the manifest's target moves: %s" % cc)
    return p


def verify_store(npz, want_ckpt_sha):
    side = sidecar_path(npz)
    if not os.path.exists(npz):
        return ["%s missing" % npz]
    if not os.path.exists(side):
        return ["%s: no provenance sidecar" % os.path.basename(npz)]
    with open(side, encoding="utf-8") as f:
        s = json.load(f)
    p = []
    if s.get("format") != SIDE_FORMAT or s.get("source") != "store":
        p.append("store sidecar format / source")
    if s.get("cold_start") is not True:
        p.append("store provenance has no cold_start true (%r): delete and embed again" % s.get("cold_start"))
    if s.get("npz_sha256") != sha_file(npz):
        p.append("store vector file sha256 differs from the sidecar")
    if s.get("ckpt_sha256") != want_ckpt_sha or npz_meta(npz)["ckpt_sha256"] != want_ckpt_sha:
        p.append("store checkpoint sha256 is not the arm's")
    return p
