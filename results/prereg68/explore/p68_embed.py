"""prereg68 (NOTES.md 9): the X2 / X3 embedding with a provenance sidecar. A thin wrapper; the frozen embedder
(p68_colab_run -> p62_run -> p61_run -> p60_run.cmd_embed -> p42_embed.main) is called unchanged.

    GPUPY results/prereg68/explore/p68_embed.py --arm X2|X3 --ckpt data/prereg62/<arm>/ckpt/step_400000.pt \
          --workers 2 --source store --players E
    GPUPY results/prereg68/explore/p68_embed.py --arm X2|X3 --ckpt ... --workers 2 --source pool \
          --src data/test_v2_download/x/tests --pool public|private

What the wrapper adds:
  * pool: p36_lib.pool_items (the function p36_embed.pool_games calls to build the item list the embedder consumes)
    is wrapped for this process; the wrapper sees the very (cands, qs) and returned item list, builds the manifest
    (p68_manifest.manifest_rows, which also asserts that its (sgf, colour) sequence equals the consumed list), and
    binds it BEFORE any part is computed: <out>.pending.json (manifest sha, checkpoint sha).
  * cold start only (audit 1122d item 2): the frozen embedder stores parts with emb / count only and reuses any part
    file present, so a part not produced in this run cannot be verified. An existing <out>.parts folder or pending
    record is refused before the embedder runs (store_E and pools alike): delete both and embed again from scratch.
  * after the embedder replaced <out>.npz: the checkpoint file is hashed again, the npz's ckpt_sha256, game count and
    per-game sample counts are compared with it / the manifest (count = the target's non-pass moves), then
    <name>.provenance.json is written atomically (tmp + fsync + replace) with the npz's sha256 inside; the pending
    record is removed.
  * an output npz that already exists without a valid sidecar is refused (no silent skip).
No score is computed. Nothing here opens the validation-answer folder.
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p68_manifest as MF  # noqa: E402

TAG = "p60_step_400000_raw_tta1"
ARMS = ("X2", "X3")


def rel(p):
    try:
        return os.path.relpath(os.path.abspath(p), REPO).replace("\\", "/")
    except ValueError:                  # another drive (tests)
        return os.path.abspath(p)


def arg(rest, flag):
    return rest[rest.index(flag) + 1] if flag in rest else None


def out_path(arm, name):
    return os.path.join(REPO, "data", "prereg62", arm, "emb", TAG, name + ".npz")


def run(argv, embed_main=None):
    """embed_main(argv) -> rc: the frozen embedder (default: p68_colab_run's p62_run.main(['embed', ...]));
    tests pass a stand-in that calls p36_embed.pool_games and writes an npz."""
    rest = list(argv)
    arm, ckpt, source = arg(rest, "--arm"), arg(rest, "--ckpt"), arg(rest, "--source")
    if arm not in ARMS or not ckpt or source not in ("store", "pool") or "--limit" in rest:
        raise SystemExit("usage: --arm X2|X3 --ckpt ... --source store|pool ... (no --limit)")
    if os.path.basename(ckpt) != "step_400000.pt":
        raise SystemExit("prereg68 embeds step_400000.pt only")
    if source == "store":
        if arg(rest, "--players") != "E":
            raise SystemExit("--source store --players E only")
        name, pool, src = "store_E", None, None
    else:
        pool, src = arg(rest, "--pool"), arg(rest, "--src")
        if pool not in ("public", "private") or not src:
            raise SystemExit("--source pool needs --src and --pool public|private")
        name = "pool_%s" % pool
    out = out_path(arm, name)
    side = MF.sidecar_path(out)
    pending = out + ".pending.json"
    ck_sha = MF.sha_file(ckpt)
    if os.path.exists(out):
        prob = (MF.verify_pool(out, src, pool, ck_sha) if source == "pool" else MF.verify_store(out, ck_sha)) \
            if os.path.exists(side) else ["%s exists without a provenance sidecar" % out]
        if prob:
            raise SystemExit("refused: %s" % prob)
        print("exists with a valid sidecar: %s" % out)
        return 0
    parts = out + ".parts"
    if os.path.exists(parts) or os.path.exists(pending):
        raise SystemExit("refused: %s or %s exists (a resumed / foreign run): delete them and embed again from "
                         "scratch (no resume in prereg68)" % (parts, pending))
    bound = {"cold_start": True}
    if source == "pool":
        import p36_lib as L36
        orig = L36.pool_items

        def hooked(cands, qs):
            res = orig(cands, qs)
            rows = MF.manifest_rows(pool, cands, qs, res[0])
            rec = {"manifest_sha256": MF.manifest_sha(rows), "ckpt_sha256": ck_sha, "pool": pool}
            if "rec" in bound:                       # a later call in this same run: must be the same list
                if rec != bound["rec"]:
                    raise SystemExit("pool_items returned another list later in the same run")
                return res
            if os.path.exists(parts) or os.path.exists(pending):
                raise SystemExit("refused: parts / pending record appeared before the embedder started")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            MF.write_json_atomic(pending, rec)
            bound["rows"] = rows
            bound["rec"] = rec
            return res

        L36.pool_items = hooked
    else:
        hooked = orig = None
    if embed_main is None:
        import p68_colab_run  # noqa: F401  (adds the X2 / X3 arms to p62_run.ARMS)
        import p62_run

        def embed_main(av):
            return p62_run.main(["embed"] + av)
    try:
        rc = embed_main(rest)
    finally:
        if orig is not None:
            import p36_lib as L36
            L36.pool_items = orig
    if rc:
        raise SystemExit("embedder rc=%s" % rc)
    if not os.path.exists(out):
        raise SystemExit("the embedder did not write %s" % out)
    if source == "pool" and "rows" not in bound:
        raise SystemExit("the hooked pool_items was never called: the manifest is not bound")
    m = MF.npz_meta(out)
    ck_sha2 = MF.sha_file(ckpt)
    prob = []
    if not (m["ckpt_sha256"] == ck_sha == ck_sha2):
        prob.append("checkpoint sha256 changed / differs from the npz")
    rec = {"format": MF.SIDE_FORMAT, "arm": arm, "name": name, "source": source, "pool": pool,
           "ckpt_path": rel(ckpt), "ckpt_sha256": ck_sha,
           "ckpt_step": m["ckpt_step"], "games": m["n"], "cold_start": True, "npz_sha256": MF.sha_file(out),
           "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    if source == "pool":
        rows = bound["rows"]
        cc = MF.count_check(m["count"], rows) if m["n"] == len(rows) else {"ok": False, "why": "length"}
        if not cc["ok"]:
            prob.append("counts vs manifest: %s" % cc)
        rec.update(src=rel(src),
                   src_csv_sha256=MF.csv_shas(src, pool), manifest_sha256=bound["rec"]["manifest_sha256"],
                   count_check=cc, manifest=rows)
    if prob:
        raise SystemExit("no sidecar written: %s" % prob)
    MF.write_json_atomic(side, rec)
    if os.path.exists(pending):
        os.remove(pending)
    print(json.dumps({k: rec[k] for k in ("arm", "name", "games", "npz_sha256", "ckpt_sha256")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
