"""prereg60 (PLAN.md Seams / Judgement / Apply): seams, the one E judgement against v2-7, official pools. CPU only.

    PY results/prereg60/explore/p60_eval.py seam [--device cpu|gpu]    # -> seam.json (both PLAN seams; c49 only)
    PY results/prereg60/explore/p60_eval.py e-use                      # prints the E use number this judgement gets
    PY results/prereg60/explore/p60_eval.py judge --e-use N            # once -> eval.json
    PY results/prereg60/explore/p60_eval.py apply --pool public|private [--seam-only]
    PY results/prereg60/explore/p60_eval.py assemble

System(vectors) = v2-7's pipeline unchanged (p51_eval.e_context / e_scores: backend cfg, wP, k of
results/prereg43/explore/tune.json c_star, P1 term) - nothing is tuned.
seam: (a) the vectors p60_run.py embed made from the widened c49 (new channel zero, early moves excluded, 200 E
  games; data/prereg60/seam[_gpu]/c49w_noearly_step_320000_raw_tta1/store_E_limit200.npz: seam = CPU fp32,
  seam_gpu = the 4090 with fp16 autocast like the stored vectors) against c49's stored vectors: same games, same
  `count`, max abs diff <= 0.05; recorded with it when the files exist: the CPU run against the frozen 17-plane
  code on the same games and device (data/prereg60/seam_ref, made by p49_run.py embed) - the direct check that the
  widening and the new game set change nothing; (b) the judging path on c49's stored E vectors == v2-7 per pool,
  bit for bit (mean 0.7035676497986656). ok = (a) of the chosen device within 0.05 and (b) identical.
judge: A60 = data/prereg60/emb/p60_step_030000_raw_tta1/store_E.npz. Adopt if mean(A60 - v2-7) >= +0.01, pool
  bootstrap (10,000, seed 43) 95% low > 0 and higher in >= 18 of 20 pools (p43_eval.judge_rule). Needs: before
  2026-11-22 12:00; no eval.json yet; PLAN.md, p60_lib.py, p60_run.py, this file, seam.json and train_record.json
  committed and unchanged; seam.json ok and made with this p60_lib.py; the training record finished with the
  30,000-step checkpoint whose sha256 is the file's and the vectors'; vectors written after the code's commit, of
  the same games / store rows as c49's file with count - c49's count in 0..7 (the early moves).
  E use number = 1 + the E judgements on record (through prereg49: p49_eval.e_judgements_so_far; later: every
  results/prereg5x|6x/explore/eval*.json that carries "e_pools_use_number" - today prereg51's, prereg54's and
  prereg57's files, i.e. p57_eval.e_uses_so_far plus prereg57's own judgement).
apply: v2-7 rebuilt on the pool == results/prereg49/apply/<pool>/top5.csv (LF bytes) first; then the same with
  prereg60's pool vectors -> results/prereg60/apply/<pool>/top5.csv. assemble: 1,600 rows, the original 800 rows
  byte-identical to v2-7's submission.
Nothing here opens the validation-answer folder or runs tools/score.py.
"""
import argparse
import datetime
import gc
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P51D = os.path.join(REPO, "results", "prereg51", "explore")
for p in (P51D,):
    if p not in sys.path:
        sys.path.insert(0, p)
import p51_eval as P51  # noqa: E402   (v2-7 records, E context, c49 files, the judging path; unchanged)

P49, E43, E45, P39E = P51.P49, P51.E43, P51.E45, P51.P39E
BK, EV, st, L = P51.BK, P51.EV, P51.st, P51.L
_rel, read_json = E43._rel, E43.read_json

DATA = os.path.join(REPO, "data", "prereg60")
PLAN = os.path.join(HERE, "PLAN.md")
LIB, RUN = os.path.join(HERE, "p60_lib.py"), os.path.join(HERE, "p60_run.py")
SEAM_JSON, EVAL_JSON, TRAIN_RECORD = (os.path.join(HERE, x) for x in ("seam.json", "eval.json", "train_record.json"))
CODE = (PLAN, LIB, RUN, os.path.abspath(__file__))
STEP = 30000
CKPT = os.path.join(DATA, "S0", "ckpt", "step_%06d.pt" % STEP)
TAG = "p60_step_%06d_raw_tta1" % STEP
VEC_DIR = os.path.join(DATA, "emb", TAG)
SEAM_NPZ = {dev: os.path.join(DATA, d, "c49w_noearly_step_320000_raw_tta1", "store_E_limit200.npz")
            for dev, d in (("cpu", "seam"), ("gpu", "seam_gpu"))}
REF_NPZ = os.path.join(DATA, "seam_ref", "V2_step_320000_raw_tta1", "store_E_limit200.npz")   # frozen code, CPU
SEAM_DEVICE = {"cpu": "cpu fp32 (the stored vectors: 4090 fp16 autocast)",
               "gpu": "4090 fp16 autocast (as the stored vectors)"}
SEAM_GAMES = 200
SEAM_TOL = 0.15   # PLAN.md revision 1
MAX_EARLY = 7                                # the target can own at most all of the first 7 non-pass moves (3 or 4 normally)
C49_SHA, V27_E, V27_APPLY = P51.C49_SHA, P51.V27_E, P51.V27_APPLY
V27_SUBMISSION = os.path.join(V27_APPLY, "submission_player.csv")
SRC_OFFICIAL = os.path.join(REPO, "data", "test_v2_download", "x", "tests")
OUT_APPLY = os.path.join(REPO, "results", "prereg60", "apply")
LAST = datetime.datetime(2026, 11, 22, 12, 0, 0)          # PLAN.md: hard stop
RAM_NEED_MB = 7000


def sha_lf(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def vec_file(which):
    return os.path.join(VEC_DIR, P51.vec_name(which))


def committed(paths):
    for f in paths:
        if not L.git_committed(f):
            raise SystemExit("%s is not committed (or differs from HEAD)" % _rel(f))


# ================================================================ E use number
def e_uses_so_far(repo=REPO):
    """E judgements on record: the prereg folders through prereg49 with an eval*.json (p49_eval's list) + every
    later results/prereg5x|6x/explore/eval*.json (not prereg60's) that carries "e_pools_use_number"."""
    prior = [d for d in P49.COMP1_E_JUDGES if glob.glob(os.path.join(repo, "results", d, "explore", "eval*.json"))]
    later = []
    for d in sorted(glob.glob(os.path.join(repo, "results", "prereg[5-9][0-9]"))):
        name = os.path.basename(d)
        if name in P49.COMP1_E_JUDGES or name == "prereg60":
            continue
        for f in sorted(glob.glob(os.path.join(d, "explore", "eval*.json"))):
            try:
                with open(f, encoding="utf-8") as fh:
                    has = "e_pools_use_number" in json.load(fh)
            except (OSError, ValueError):
                raise SystemExit("cannot read %s" % f)
            if has:
                later.append("%s/%s" % (name, os.path.basename(f)))
    return {"through_prereg49": prior, "later": later, "count": len(prior) + len(later)}


# ================================================================ checks of the vectors
def check_vectors(path, ref, ckpt_sha, tag=TAG):
    """prereg60 vectors: made from the given checkpoint (raw, tta 1, all samples incl. early moves, action plane),
    for exactly the games of c49's file; count - c49's count in 0..MAX_EARLY. Returns a record."""
    if not os.path.exists(path):
        raise SystemExit("missing %s" % _rel(path))
    problems = []
    cfg = E43.embed_cfg_of(path) or {}
    rcfg = E43.embed_cfg_of(ref) or {}
    if cfg.get("tta") != 1 or cfg.get("weights") != "raw":
        problems.append("tta %r weights %r" % (cfg.get("tta"), cfg.get("weights")))
    if (cfg.get("games"), cfg.get("moves")) != (rcfg.get("games"), rcfg.get("moves")):
        problems.append("games / moves differ from c49's file")
    with np.load(path, allow_pickle=False) as z, np.load(ref, allow_pickle=False) as r:
        if str(z["ckpt_sha256"]) != ckpt_sha or cfg.get("ckpt_sha256") != ckpt_sha:
            problems.append("checkpoint sha %s != %s" % (str(z["ckpt_sha256"]), ckpt_sha))
        if str(z["ckpt"]) != tag:
            problems.append("tag %r != %r (debugging switches / another step?)" % (str(z["ckpt"]), tag))
        if int(z["ckpt_step"]) != STEP:
            problems.append("checkpoint step %s" % z["ckpt_step"])
        if z["emb"].shape != r["emb"].shape:
            problems.append("shape %s != c49's %s" % (z["emb"].shape, r["emb"].shape))
        else:
            d = z["count"].astype(np.int64) - r["count"].astype(np.int64)
            if d.min() < 0 or d.max() > MAX_EARLY:
                problems.append("count - c49's count spans %d..%d, expected 0..%d" % (d.min(), d.max(), MAX_EARLY))
            extra = {"more_samples_total": int(d.sum()), "more_samples_per_game_mean": float(d.mean()),
                     "games_with_vector": int((z["count"] > 0).sum()),
                     "games_with_vector_c49": int((r["count"] > 0).sum())}
        for k in ("store_row", "players_sha256_sorted_ids"):
            if (k in z.files) != (k in r.files) or (k in z.files and not np.array_equal(z[k], r[k])):
                problems.append("%s differs from c49's file" % k)
    if problems:
        raise SystemExit("%s: %s" % (_rel(path), "; ".join(problems)))
    out = {"file": _rel(path), "file_sha256": L.sha256_file(path), "ckpt": _rel(CKPT), "ckpt_sha256": ckpt_sha,
           "embed_cfg": cfg, "same_games_as": _rel(ref)}
    out.update(extra)
    return out


def train_record(path=TRAIN_RECORD, ckpt=CKPT):
    """The committed training record: finished, 30,000 steps, and the checkpoint file is the recorded one."""
    committed([path])
    tr = read_json(path)
    name = os.path.basename(ckpt)
    if not tr.get("finished") or tr.get("steps") != STEP or tr.get("smoke") or name not in tr.get("checkpoints", {}):
        raise SystemExit("%s: not a finished %d-step run" % (_rel(path), STEP))
    if not os.path.exists(ckpt):
        raise SystemExit("missing %s" % _rel(ckpt))
    sha = L.sha256_file(ckpt)
    if sha != tr["checkpoints"][name]:
        raise SystemExit("%s is not the checkpoint of the training record" % _rel(ckpt))
    return tr, sha


def seam_ok(path=SEAM_JSON):
    committed([path])
    s = read_json(path)
    if not s.get("ok"):
        raise SystemExit("seam.json is not ok")
    if s["code_sha256"]["p60_lib.py"] != sha_lf(LIB):
        raise SystemExit("seam.json was made with another p60_lib.py: run the seam again")
    return s


# ================================================================ seam
def _stats(x, ref):
    """Differences of two sets of game vectors (rows = games)."""
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    diff = np.abs(x - ref)
    nx, nr = np.linalg.norm(x, axis=1), np.linalg.norm(ref, axis=1)
    cos = (x * ref).sum(1) / np.maximum(nx * nr, 1e-30)
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "largest_abs_value": float(np.abs(ref).max()), "median_norm": float(np.median(nr)),
            "max_abs_diff_over_median_norm": float(diff.max() / np.median(nr)), "min_cosine": float(cos.min())}


def seam_a(device="cpu"):
    """PLAN seam 1 from the file p60_run.py embed wrote (widened c49, --no-early, the first 200 E games)."""
    npz = SEAM_NPZ[device]
    if not os.path.exists(npz):
        raise SystemExit("missing %s: run p60_run.py embed (see its docstring, seam (a)) first" % _rel(npz))
    ref = P51.c49_file("E")
    with np.load(npz, allow_pickle=False) as z, np.load(ref, allow_pickle=False) as r:
        n = len(z["emb"])
        cfg = json.loads(str(z["embed_cfg"]))
        rec = {"file": _rel(npz), "file_sha256": L.sha256_file(npz), "reference": _rel(ref),
               "reference_sha256": L.sha256_file(ref), "ckpt_sha256": str(z["ckpt_sha256"]), "tag": str(z["ckpt"]),
               "games": int(n), "games_with_sample": int((z["count"] > 0).sum()),
               "positions": int(z["count"].sum()), "weights": cfg.get("weights"), "tta": cfg.get("tta"),
               "store_row_equal": bool(np.array_equal(z["store_row"], r["store_row"][:n])),
               "count_equal": bool(np.array_equal(z["count"], r["count"][:n]))}
        rec.update(_stats(z["emb"], r["emb"][:n]))
    rec.update({"tolerance": SEAM_TOL, "device": SEAM_DEVICE[device]})
    rec["ok"] = bool(rec["ckpt_sha256"] == C49_SHA and rec["tag"] == "c49w_noearly_step_320000_raw_tta1" and
                     n == SEAM_GAMES and rec["weights"] == "raw" and rec["tta"] == 1 and rec["store_row_equal"] and
                     rec["count_equal"] and rec["max_abs_diff"] <= SEAM_TOL)
    return rec


def seam_same_device():
    """The CPU run of the widened c49 against the frozen 17-plane code (p49_run.py embed) on the same 200 games
    and the same device: no fp16 / device noise in between. None when a file is missing."""
    a, b = SEAM_NPZ["cpu"], REF_NPZ
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    with np.load(a, allow_pickle=False) as z, np.load(b, allow_pickle=False) as r:
        rec = {"file": _rel(a), "file_sha256": L.sha256_file(a), "reference": _rel(b),
               "reference_sha256": L.sha256_file(b), "reference_tag": str(r["ckpt"]),
               "reference_ckpt_sha256": str(r["ckpt_sha256"]), "games": int(len(z["emb"])),
               "store_row_equal": bool(np.array_equal(z["store_row"], r["store_row"])),
               "count_equal": bool(np.array_equal(z["count"], r["count"])),
               "device": "both on CPU fp32: widened 18-plane network + p60_lib.GameSet60 (--no-early) against the "
                         "frozen p42_embed path (17 planes)"}
        rec.update(_stats(z["emb"], r["emb"]))
    rec["ok"] = bool(rec["reference_ckpt_sha256"] == C49_SHA and rec["store_row_equal"] and rec["count_equal"] and
                     rec["max_abs_diff"] <= 1e-3)
    return rec


def cmd_seam(a):
    t0 = time.time()
    ra = seam_a(a.device)
    other = "gpu" if a.device == "cpu" else "cpu"
    ro = seam_a(other) if os.path.exists(SEAM_NPZ[other]) else None
    rs = seam_same_device()
    pools, zs, _sets, v26 = P51.e_context()
    per, info = P51.base_seam(P51.v27_base(), pools, zs, v26)          # raises unless bit for bit
    rb = {"rule": "the judging path on c49's stored E vectors == v2-7 per pool, bit for bit",
          "mean": float(np.mean(per)), "want_mean": V27_E, "identical": bool(float(np.mean(per)) == V27_E),
          "pools": len(per), "vectors": info}
    rec = {"format": "c10-prereg60 seam v2",
           "rule": "ok = a (same games, same count, max abs diff <= %.2f) and b (bit for bit)" % SEAM_TOL,
           "a_device": a.device, "a_widened_c49_equals_c49": ra, "a_other_device": ro,
           "a_same_device_vs_frozen_code": rs, "b_v27_E_bit_for_bit": rb,
           "ok": bool(ra["ok"] and rb["identical"]),
           "code_sha256": {os.path.basename(f): sha_lf(f) for f in (LIB, RUN, os.path.abspath(__file__))},
           "git_head": L.git_head(), "peak_working_set_mb": E45.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(a.out, rec)
    print(json.dumps({"a": {k: ra[k] for k in ("device", "games", "count_equal", "max_abs_diff", "ok")},
                      "same_device": None if rs is None else {k: rs[k] for k in ("max_abs_diff", "ok")},
                      "b": {k: rb[k] for k in ("mean", "identical")}, "ok": rec["ok"]}))
    return 0 if rec["ok"] else 1


# ================================================================ judge (E, once)
def cmd_judge(a):
    t0 = time.time()
    if datetime.datetime.now() > LAST:
        raise SystemExit("past %s (PLAN.md hard stop): dropped" % LAST)
    if os.path.exists(EVAL_JSON):
        raise SystemExit("%s exists: the judgement is made once" % _rel(EVAL_JSON))
    committed(CODE)
    seam = seam_ok()
    tr, sha = train_record()
    uses = e_uses_so_far()
    if a.e_use != uses["count"] + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before (%s)" % (a.e_use, uses["count"],
                                                                                         uses))
    path = vec_file("E")
    vec = check_vectors(path, P51.c49_file("E"), sha)
    E43.made_after([path], max(E43.commit_time(f) for f in CODE))
    pools, zs, sets, v26 = P51.e_context()
    base = P51.v27_base()
    bper, binfo = P51.base_seam(base, pools, zs, v26)
    per, byk = P51.e_scores(path, pools, zs, v26)
    j = E43.judge_rule(per, bper)
    rec = {"format": "c10-prereg60 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": uses, "e_reused": True, "development_set": None,
           "vectors": vec, "train_record_sha256": sha_lf(TRAIN_RECORD), "train_seconds": tr.get("seconds"),
           "seam_sha256": sha_lf(SEAM_JSON), "seam_a_device": seam["a_device"],
           "seam_a_max_abs_diff": seam["a_widened_c49_equals_c49"]["max_abs_diff"],
           "base": {"name": "c49 (v2-7)", "eval": base["eval"], "e_mean": base["e_mean"], "vectors_E": binfo},
           "cfg": v26["cfg"], "wP": v26["wP"], "k": v26["k"], "seam_base_E_identical": True,
           "git_head": L.git_head(), "p1_sets": sets,
           "means": {"p60": float(np.mean(per)), "base": float(np.mean(bper))},
           "per_pool": {"p60": per, "base": list(bper)}, "by_k": {"p60": EV.mean_by_k(byk)}}
    rec["p60_minus_base"] = {x: j[x] for x in ("mean", "ci95", "pools_higher")}
    rec["rule"] = j["rule"].replace("c*", "A60")
    rec["adopt"] = j["adopt"]
    rec["peak_working_set_mb"] = E45.peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(EVAL_JSON, rec)
    print(json.dumps({x: rec[x] for x in ("means", "p60_minus_base", "by_k", "adopt", "e_pools_use_number")},
                     indent=1))
    return 0


# ================================================================ apply / assemble
def lf_bytes(path):
    """(the file's bytes with CRLF -> LF, whether it had CR): the committed v2-7 files are LF; a checkout with
    core.autocrlf=true gives CRLF working copies of the same content (as p57_eval.lf_bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    return data.replace(b"\r\n", b"\n"), b"\r" in data


def original_rows(data):
    """The lines of the original 800 questions (ids pub_ / priv_) of a submission, as bytes."""
    return [x for x in data.split(b"\n")[1:] if x.startswith((b"pub_", b"priv_"))]


def assemble_check(new, old):
    """PLAN.md Apply: 1,600 rows, the original 800 rows byte-identical to v2-7's submission (LF bytes of both)."""
    rows = [x for x in new.split(b"\n")[1:] if x]
    a, b = original_rows(new), original_rows(old)
    chk = {"rows": len(rows), "original_rows": len(a), "new_rows": len(rows) - len(a),
           "header_identical_to_v2_7": new.split(b"\n")[0] == old.split(b"\n")[0],
           "original_rows_identical_to_v2_7": a == b}
    chk["ok"] = bool(chk["rows"] == 1600 and chk["original_rows"] == 800 and chk["new_rows"] == 800 and
                     chk["header_identical_to_v2_7"] and chk["original_rows_identical_to_v2_7"])
    return chk


def cmd_apply(a):
    t0 = time.time()
    sha = None
    if not a.seam_only:
        committed([EVAL_JSON])
        if not read_json(EVAL_JSON).get("adopt"):
            raise SystemExit("eval.json does not adopt: keep v2-7")
        _tr, sha = train_record()
    v26 = E45.v26()
    cfg, wP, k = v26["cfg"], v26["wP"], v26["k"]
    rec, items, cand_ids, qids = E43.official_rec(a.src, a.pool)
    name = {"public": "pub2", "private": "priv2"}[a.pool]
    Pset, Lset = P39E.load_set(name), P39E.load_set("L")
    if len(Pset["games"]) != len(items) or any(g["color"] != c for g, (_s, c) in zip(Pset["games"], items)):
        raise SystemExit("the P1 list of the pool does not match the pool's games")
    if Pset["meta"] != Lset["meta"]:
        raise SystemExit("L and pool chunks have different metadata")
    bad = int((~Pset["usable"]).sum())
    if bad > P39E.FAIL_MAX * len(items):
        raise SystemExit("%d of %d pool games have no usable P1 (> 0.5%%): stop" % (bad, len(items)))
    models = P39E.fit_models(Lset)
    del Lset
    if P39E.models_sha(models) != v26["lda_sha256"]:
        raise SystemExit("the L LDA refit differs from prereg43 tune.json")
    colour = np.array([c for _s, c in items])
    z, has = P39E.p1_terms(Pset["X"], Pset["usable"], colour, rec["cand_rows"], rec["q_rows"], models,
                           E43.P1_PW)[E43.P1_VARIANT]
    del models, Pset
    gc.collect()

    def top5_of(e):
        F = BK.fuse_cfg(rec, BK.e_term(rec, e, cfg), cfg)
        return P39E.top5_bytes(qids, st.top5(st.csls(F + wP * z, k), cand_ids))

    bpath = P51.c49_file(a.pool)
    eb = E43.pool_emb(bpath, items, C49_SHA)
    data_b = top5_of(eb)
    del eb
    bfile = os.path.join(V27_APPLY, a.pool, "top5.csv")
    want_b, crlf = lf_bytes(bfile)
    if want_b != data_b:
        raise SystemExit("seam: v2-7 rebuilt on %s differs from %s" % (a.pool, _rel(bfile)))
    if a.seam_only:
        print(json.dumps({"pool": a.pool, "v27_top5_identical": True, "file": _rel(bfile),
                          "working_copy_had_crlf": crlf, "questions": len(qids), "candidates": len(cand_ids),
                          "games": len(items), "peak_working_set_mb": E45.peak_mb(),
                          "seconds": round(time.time() - t0, 1)}))
        return 0
    path = vec_file(a.pool)
    info = check_vectors(path, bpath, sha)
    E43.made_after([path], max(E43.commit_time(f) for f in CODE))
    e = E43.pool_emb(path, items, sha)
    data = top5_of(e)
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as fh:
        fh.write(data)
    G = BK.pool_games(e, rec)
    rep = {"format": "c10-prereg60 apply v1", "pool": a.pool, "vectors": info, "cfg": cfg, "wP": wP, "k": k,
           "base": "c49 (v2-7)", "base_top5_equals": _rel(bfile), "base_working_copy_had_crlf": crlf,
           "questions": len(qids), "candidates": len(cand_ids), "games": len(items),
           "games_without_vector": int((e.count == 0).sum()),
           "candidates_without_vector": int(G["nc"] - len(np.unique(G["gc"]))),
           "questions_without_vector": int(G["nq"] - len(np.unique(G["gq"]))),
           "games_without_P1": bad, "questions_without_P1": int((~np.asarray(has)).sum()),
           "eval_sha256": sha_lf(EVAL_JSON), "top5_sha256": L.sha256_file(out),
           "differs_from_base_rows": int(sum(x != y for x, y in zip(data.split(b"\n"), data_b.split(b"\n")))),
           "peak_working_set_mb": E45.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    if not read_json(EVAL_JSON).get("adopt"):
        raise SystemExit("eval.json does not adopt: keep v2-7")
    rc = E43.cmd_assemble(a)          # p35_apply.cmd_assemble: ids, prefixes, top-5 of the pool's candidates
    with open(os.path.join(a.out, "submission_player.csv"), "rb") as f:
        new = f.read()
    old, crlf = lf_bytes(V27_SUBMISSION)
    chk = assemble_check(new, old)
    chk.update({"v2_7_working_copy_had_crlf": crlf, "v2_7_submission": _rel(V27_SUBMISSION),
                "v2_7_sha256": L.sha256_file(V27_SUBMISSION),
                "sha256": L.sha256_file(os.path.join(a.out, "submission_player.csv")), "assemble_rc": rc})
    chk["ok"] = bool(chk["ok"] and rc == 0)
    L.write_json(os.path.join(a.out, "assemble_check.json"), chk)
    print(json.dumps(chk, indent=1))
    return 0 if chk["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("seam")
    m.add_argument("--out", default=SEAM_JSON)
    m.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                   help="which run of seam (a) decides: data/prereg60/seam (CPU fp32) or seam_gpu (4090 fp16)")
    sub.add_parser("e-use")
    j = sub.add_parser("judge")
    j.add_argument("--e-use", type=int, required=True)
    p = sub.add_parser("apply")
    p.add_argument("--src", default=SRC_OFFICIAL)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--seam-only", action="store_true", help="only rebuild v2-7 on the pool and compare (no eval.json)")
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", default=P49.V1_SUBMISSION)
    b.add_argument("--src", default=SRC_OFFICIAL)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd == "e-use":
        print(e_uses_so_far()["count"] + 1)
        return 0
    if a.cmd != "assemble":
        BK.polite()
        BK.wait_for_ram(need_mb=RAM_NEED_MB)
    return {"seam": cmd_seam, "judge": cmd_judge, "apply": cmd_apply, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
