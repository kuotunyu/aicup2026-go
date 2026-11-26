"""prereg17 exploration (competition 2): TabICLv2 as a 7th member of Q09's ensemble, inner-only judgement of PLAN.md
(one candidate Q09+T), judged once. CPU (.venv); the TabICL probabilities come from p17_tabicl.py (GPU, own env).

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg17/explore
    $S/p17_eval.py eval            -> $S/eval_report.json (judged once; the code must be committed and clean)
    $S/p17_eval.py eval --smoke    -> data/prereg17/smoke/eval_report.json (noise features, 1,000-game context:
                                      pipeline and timing only, never a result)

Inputs (read only): data/prereg17/prep (09's prep rebuilt; seam P), data/prereg17/feat or smoke/feat (feat_record.json),
data/prereg17/tabicl or smoke/tabicl (run_record.json + chunks), data/prereg12/q09_inner/ (ft_eval.InnerPrep), Q09's
frozen p1e_map.npz, u1_lib's inner pseudo-questions through tools/p09_rank_fuse.Inner. Never data/local, never
validation features, never tools/score.py.
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
FT_DIR = os.path.join(REPO, "results", "prereg12", "explore", "finetune")
for p in (TOOLS, FT_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p09_rank_fuse as r  # noqa: E402  (pins the p08 tools before importing them)
import p09_forward as FW  # noqa: E402
import ft_eval  # noqa: E402  (prereg12: InnerPrep, guard_path)
import p17_lib as K  # noqa: E402
import p17_prep as PP  # noqa: E402
import p17_tabicl as TT  # noqa: E402  (constants only; no torch / tabicl at import)

TOL = K.TOL
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
Q09_TUNE = os.path.join(REPO, "results", "prereg09", "rank", "tune_report.json")
Q09_FROZEN = os.path.join(REPO, "results", "prereg09", "rank", "frozen_config.json")
PREP_DIR = os.path.join(REPO, "data", "prereg12", "q09_inner")
DATA = os.path.join(REPO, "data", "prereg17")
SMOKE_DIR = os.path.join(DATA, "smoke")
STARTED = os.path.join(DATA, "eval_started.json")
REPORT = os.path.join(HERE, "eval_report.json")
OWN_FILES = ("PLAN.md", "p17_lib.py", "p17_prep.py", "p17_tabicl.py", "run_p17_tabicl.sh", "p17_eval.py",
             "test_p17.py")
IMPORTED = ("tools/p02_rank_fuse.py", "tools/p03_rank_fuse.py", "tools/p04_rank_fuse.py", "tools/p07_rank_fuse.py",
            "tools/p08_rank_fuse.py", "tools/p09_rank_fuse.py", "tools/p09_forward.py", "tools/p09_map.py",
            "results/prereg12/explore/finetune/ft_eval.py", "results/prereg05/explore/u1/u1_lib.py",
            "results/prereg07/explore/pilot_humansl/pilot_fuse.py", "results/prereg07/explore/pilot_humansl/hs_lib.py",
            "results/prereg08/p1/p1_fuse.py", "results/prereg08/p1/p1_lib.py",
            "tools/p02_rank_pseudo.py", "tools/p02_rank_opening.py", "tools/p02_moves.py",
            "results/prereg09/rank/p1e_map.npz", "results/prereg09/rank/check_report.json",
            "results/prereg09/rank/tune_report.json", "results/prereg09/rank/frozen_config.json",
            "results/prereg08/rank/p1_map.npz", "results/prereg08/feat/inner_manifest.json",
            "results/prereg09/feat/p3_manifest.json")
FORMAT = "c10-prereg17 explore eval v1"
guard_path = ft_eval.guard_path


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def sha_lf(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def code_state():
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN_FILES] + list(IMPORTED)
    dirty = git("status", "--porcelain", "--", *rel)
    return {"head": git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def once_guard(a, cs, out_path):
    """Judged once: a report or a started marker (an aborted run) needs --rerun --reason and a newer HEAD."""
    prior = [p for p in (out_path, STARTED) if os.path.exists(p)]
    if not prior:
        return None
    if not (a.rerun and a.reason):
        raise SystemExit(f"{prior} exist: judged once (--rerun --reason TEXT only after a commit that names a "
                         f"concrete program error)")
    heads = set()
    for p in prior:
        with open(p, encoding="utf-8") as f:
            heads.add(((json.load(f).get("code") or {}).get("head")))
    if cs["head"] in heads:
        raise SystemExit("--rerun needs a newer commit than the earlier run's (the fix of the named error)")
    return sorted(h for h in heads if h)


def tune_ref():
    with open(Q09_TUNE, encoding="utf-8") as f:
        c = json.load(f)["combinations"]["Q09"]
    return {"Q09": c["oof"]["rank_balanced"], "Q09_mono": c["single_colour"]["rank_balanced"]}


# ---------------------------------------------------------------- inputs of the 7th member
def load_feat(feat_dir, prep_inner_ref, smoke):
    """feat_record.json + file SHA + inner_ref equal to the prepared inner rows + the context re-derived ->
    (record, problems)."""
    frec = r.read_json(os.path.join(feat_dir, "feat_record.json"))
    probs = []
    if frec.get("status") != "ok":
        probs.append(f"feat_record status {frec.get('status')!r}")
    if bool(frec.get("smoke")) != bool(smoke):
        probs.append(f"feat_record smoke {frec.get('smoke')!r} in a {'smoke' if smoke else 'real'} run")
    y_probe = r.Prep(PP.PREP).y
    ctx = K.stratified_context(y_probe, K.CONTEXT_PER_LEVEL, K.CONTEXT_SEED)
    if smoke:
        ctx = K.smoke_context(ctx, y_probe)
    if not (np.array_equal(np.load(os.path.join(feat_dir, "ctx_rows.npy")), ctx)
            and np.array_equal(np.load(os.path.join(feat_dir, "y_ctx.npy")), y_probe[ctx])):
        probs.append("ctx_rows.npy / y_ctx.npy differ from the re-derived context (PLAN.md 2.1 item 4)")
    for f, want in (frec.get("files_sha256") or {}).items():
        if PP.sha_file(os.path.join(feat_dir, f)) != want:
            probs.append(f"{f}: SHA-256 differs from feat_record.json")
    iref = np.load(os.path.join(feat_dir, "inner_ref.npy"))
    if not np.array_equal(iref, np.asarray(prep_inner_ref)):
        probs.append("inner_ref.npy differs from the prepared inner rows used by Inner")
    pca = frec.get("pca") or {}
    if pca.get("rows") != 270000 or pca.get("dim") != K.PCA_DIM or pca.get("sources") != sorted(r.PROBE_SETS):
        probs.append(f"PCA record differs from PLAN.md 2.1: {pca}")
    return frec, probs


def seam_c(inner, feat_dir, smoke):
    players = np.load(os.path.join(feat_dir, "ctx_players.npy"))
    keys = inner.d["inner"]["keys"]
    full, ids = K.player_overlap(players, keys["rank"], keys["player_id"])
    y = np.load(os.path.join(feat_dir, "y_ctx.npy"))
    want = [K.SMOKE_PER_LEVEL if smoke else K.CONTEXT_PER_LEVEL] * K.L
    per = [int((y == k).sum()) for k in range(K.L)]
    return {"ok": bool(full == 0 and per == want), "overlap_rank_player": int(full),
            "overlap_player_id_only_record": int(ids), "context_players": int(len(set(players.tolist()))),
            "context_per_level": per}


def load_tabicl(tab_dir, frec, n_rows):
    """run_record.json + chunks -> (p_T [n, 10] float64 row-normalised, seam T record)."""
    rec = r.read_json(os.path.join(tab_dir, "run_record.json"))
    probs = []
    want_inputs = {k: frec["files_sha256"][k] for k in TT.INPUTS}
    want_inputs["ckpt"] = K.CKPT_SHA256
    if rec.get("inputs_sha256") != want_inputs:
        probs.append("run_record inputs differ from feat_record.json / the registered checkpoint")
    if rec.get("params") != K.TABICL_PARAMS or TT.TABICL_PARAMS != K.TABICL_PARAMS:
        probs.append("TabICL parameters differ from PLAN.md 2.2")
    pk = (rec.get("versions") or {}).get("packages") or {}
    if pk.get("tabicl") != K.TABICL_VERSION or pk.get("torch") != K.TORCH_VERSION:
        probs.append(f"versions differ: tabicl {pk.get('tabicl')}, torch {pk.get('torch')}")
    if rec.get("classes") != list(range(K.L)):
        probs.append(f"classes {rec.get('classes')}")
    if rec.get("chunk") != K.CHUNK or not rec.get("complete") or rec.get("rows") != n_rows:
        probs.append("run not complete or chunking differs")
    recs = []
    for s, e in K.chunk_bounds(n_rows):
        p = TT.chunk_path(tab_dir, s)
        name = os.path.basename(p)
        if (rec.get("chunk_sha256") or {}).get(name) != PP.sha_file(p):
            probs.append(f"{name}: SHA-256 differs from run_record.json")
            continue
        with np.load(p, allow_pickle=False) as z:
            if json.loads(str(z["inputs_sha256"])) != want_inputs:
                probs.append(f"{name}: inputs differ")
            recs.append((int(z["start"]), int(z["end"]), np.asarray(z["prob"])))
    out = {"problems": probs, "chunks": len(recs), "gpu_seconds_total": rec.get("gpu_seconds_total"),
           "sessions": rec.get("sessions"), "versions": {k: pk.get(k) for k in ("tabicl", "torch", "numpy", "scipy",
                                                                                   "scikit-learn")}}
    if probs:
        out["ok"] = False
        return None, out
    raw = K.assemble_chunks(recs, n_rows)
    finite = bool(np.all(np.isfinite(raw)) and np.all(raw >= 0))
    p_t, dev = K.normalise_rows(raw)
    out.update(raw_dtype=str(raw.dtype), finite_nonnegative=finite, row_sum_max_abs_dev=dev,
               ok=bool(finite and np.all(raw.sum(axis=1) > 0)))
    return (p_t if out["ok"] else None), out


# ---------------------------------------------------------------- eval
def cmd_eval(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_path = os.path.join(SMOKE_DIR, "eval_report.json") if smoke else REPORT
    feat_dir = os.path.join(SMOKE_DIR, "feat") if smoke else PP.FEAT
    tab_dir = os.path.join(SMOKE_DIR, "tabicl") if smoke else os.path.join(DATA, "tabicl")
    cs = code_state()
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg17 files or the imported tools are not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, out_path)
        write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                             "rerun_reason": a.reason if a.rerun else None})
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    prep = ft_eval.InnerPrep(guard_path(os.path.abspath(a.prep_dir)))
    map_sha = r.sha_raw(Q09_MAP)
    with open(Q09_FROZEN, encoding="utf-8") as f:
        if map_sha != json.load(f)["map"]["sha256"]:
            raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    inner = r.Inner(prep, log=log)
    ref = tune_ref()
    mp = FW.load_map(Q09_MAP)
    lp_q09 = r.inner_map_lp(prep, mp)
    _lp, members = r.inner_map_lp(prep, mp, with_members=True)
    E_q09, _ = inner.E_p1(lp_q09)
    terms9 = list(r.Q08_TERMS)
    q09 = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E_q09})
    base = {"format": FORMAT, "smoke": smoke, "code": cs}
    # ---- seam 1
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]]}
    seam1 = all(abs(x - y) <= TOL for x, y in seam.values())
    # ---- seam M: the six members reproduce Q09's lp bit for bit
    seam_m = {"members": len(members), "lse_mean_equal": bool(np.array_equal(K.lse_mean(members), lp_q09)),
              "with_members_lp_equal": bool(np.array_equal(_lp, lp_q09))}
    seam_m["ok"] = bool(len(members) == 6 and seam_m["lse_mean_equal"] and seam_m["with_members_lp_equal"])
    # ---- seam P: 09's prep rebuilt byte for byte (re-hashed now) and Inner's inner_X is the same file content
    seam_p = PP.seam_p()
    log(f"seam 1 {'ok' if seam1 else 'FAILED'}: {seam}; seam M {seam_m}; seam P {seam_p['ok']}")
    if not (seam1 and seam_m["ok"] and seam_p["ok"]):
        write_json(out_path, {**base, "status": "stop: seam failed", "seam": seam, "seam_m": seam_m,
                              "seam_p": seam_p})
        return 1
    # ---- inputs of the 7th member: feat (+ seam C), TabICL chunks (seam T)
    frec, fprobs = load_feat(feat_dir, prep.inner_ref, smoke)
    sc = seam_c(inner, feat_dir, smoke)
    p_t, seam_t = load_tabicl(tab_dir, frec, len(prep.inner_ref))
    log(f"feat problems {fprobs}; seam C {sc}; seam T ok {seam_t.get('ok')} {seam_t['problems']}")
    if fprobs or not sc["ok"] or p_t is None:
        write_json(out_path, {**base, "status": "stop: seam failed", "seam": seam, "seam_m": seam_m,
                              "seam_p": seam_p, "feat_problems": fprobs, "seam_c": sc, "seam_t": seam_t})
        return 1
    # ---- candidate Q09+T
    tc = time.time()
    lp7 = K.mean7_lp(members, p_t)
    E7, _info = inner.E_p1(lp7)
    cand = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E7})
    cmp_ = inner.compare(cand, q09)
    g = K.gates(cmp_)
    log(f"Q09+T {cand['rb']!r} vs Q09 {q09['rb']!r}: {json.dumps(g)} ({time.time() - tc:.0f}s)")
    # ---- record only
    t_alone = inner.evaluate(np.log(np.clip(p_t, K.LSE_FLOOR, None)))
    rec_only = {"T_alone_x4_in_P1TO_slot": inner.short(t_alone, q09),
                "pca_explained_ratio_128": (frec.get("pca") or {}).get("explained_ratio"),
                "context": frec.get("context"), "gpu_seconds_total": seam_t.get("gpu_seconds_total"),
                "p_T_row_sum_max_abs_dev": seam_t.get("row_sum_max_abs_dev")}
    status = "gates passed (worth a PREREG_17; not adoption)" if g["passed"] else "gates not passed: stop"
    rep = {**base, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke_note": ("features replaced by default_rng(12345) normal noise, 1,000-game context: numbers are "
                          "meaningless" if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md: inner-only judgement of Q09+T (TabICLv2 on PCA-128 of the 1,540 P1 features, 30,000 probe "
                   "games in context, as a 7th equal-weight member of Q09's ensemble, fused as Q09); training players "
                   "only, no validation scoring (competition 2 has none left)",
           "status": status, "seam": seam, "seam_m": seam_m, "seam_p": seam_p, "seam_c": sc, "seam_t": seam_t,
           "feat_record": {k: frec.get(k) for k in ("head", "pca", "context", "inner_rows", "files_sha256")},
           "candidate": {"name": "Q09+T", "terms": cand["terms"], **inner.short(cand, q09)},
           "baseline": {"name": "Q09", **inner.short(q09)}, "gates": g, "record_only": rec_only,
           "q09_map_sha256": map_sha}
    write_json(out_path, rep)
    log(f"-> {out_path}: {status}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--prep-dir", default=PREP_DIR)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
