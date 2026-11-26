"""prereg59 (competition 2): opponent-identity term, second version, added to Q25 (PLAN.md; NOTES.md for the narrow
readings). CPU only, .venv (numpy / scipy), 4 cores, working set <= 3 GB (p58_lib.limits). prereg58's code
(p58_lib, p58_eval: the lean Q25 harness, the HD join, the vector loader) is imported, never edited.

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8; no OMP / OPENBLAS / MKL variable
    S=results/prereg59/explore;  PY=.venv/Scripts/python.exe
    $PY $S/p59_eval.py rows      -> data/prereg59/val_rows.npy; $S/rows.json  (validation players' rows, no model number)
    (GPU, later: data/prereg59/embed_val.sh -> data/prereg59/emb_own_val/...)
    $PY $S/p59_eval.py kernel    -> $S/kernel.json, data/prereg59/prototypes.npz   (M, tau1, tau2, b; chosen once)
    $PY $S/p59_eval.py seam      -> $S/seam.json   (Q25 on inner through prereg58's path, bit for bit)
    $PY $S/p59_eval.py eval      -> $S/eval.json   (judged once: kernel.json and seam.json committed, tree clean)
    $PY $S/p59_eval.py results   -> $S/RESULTS.md  (numbers of eval.json / kernel.json, written mechanically)
    $PY $S/p59_eval.py summary kernel|seam|eval    one line for the chain's commit messages (seam: exit 1 if not ok)
  checks that produce no result:
    $PY $S/p59_eval.py kernel --dry-de --out-dir DIR   the kernel code path with the D / E prototypes only (no
                                                       validation vectors), written to DIR outside results/
    $PY $S/p59_eval.py eval --smoke --kernel-dir DIR   the judgement code path with the inner opponent vectors, HD
                                                       groups and colours replaced by default_rng(12345) draws
                                                       -> data/prereg59/smoke/ (DIR under data/, not a temp folder:
                                                       the harness's guard refuses paths containing "local")
    $PY $S/p59_eval.py results --from-dir D --out F    the RESULTS writer on such files

No inner number of the new term exists before `eval`. Never data/local, never tools/score.py, never a test-set file
(p58_lib.guard), no public score.
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P58 = os.path.join(REPO, "results", "prereg58", "explore")
P36 = os.path.join(REPO, "results", "prereg36", "explore")
P49 = os.path.join(REPO, "results", "prereg49", "explore")
for p in (P58, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p58_eval as E58  # noqa: E402  (puts tools, prereg23, prereg41 on sys.path; imports p58_lib)
import p58_lib as K58  # noqa: E402
import p59_lib as K  # noqa: E402

ROWS_JSON = os.path.join(HERE, "rows.json")
SMOKE_DIR = os.path.join(K.DATA, "smoke")
TERM = "OPP2"
NS = (5, 10, 20)
DRAWS = 20
SEED = 59
NOISE_SEED = 12345
HARD_STOP = "2026-11-23T12:00:00"       # PLAN.md
log, git, rel, committed_clean, peak_gb = E58.log, E58.git, E58.rel, E58.committed_clean, E58.peak_gb


def code_state():
    mods = E58.loaded_repo_modules()
    return {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "prereg59_dirty": git("status", "--porcelain", "--", "results/prereg59"),
            "imported_dirty": git("status", "--porcelain", "--", *mods) if mods else "",
            "sha256_lf": {x: K58.sha_lf(os.path.join(REPO, *x.split("/"))) for x in mods}}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def past_hard_stop():
    return now() >= HARD_STOP


# ================================================================ the validation players (committed split file only)
def split_players():
    """(inner ids, validation ids) of split v1 exactly as prereg49 read them (p49_lib.extra_players: the committed
    results/split_v1/players_split.csv through tools/c10data.read_split)."""
    for p in (P49, P36):
        if p not in sys.path:
            sys.path.insert(0, p)
    import p49_lib as L49
    inner, val = L49.extra_players()
    return set(inner), set(val), L49


def cmd_rows(a):
    st = K58.limits(log)
    if os.path.exists(ROWS_JSON):
        raise SystemExit("rows.json exists")
    inner, val, L49 = split_players()
    cache = L49.Cache49()
    pid = np.asarray(cache.pid).astype(str)
    rows = np.flatnonzero(np.isin(pid, np.array(sorted(val)))).astype(np.int64)
    rk = np.asarray(cache.rank).astype(str)[rows]
    u, first, n = np.unique(pid[rows], return_index=True, return_counts=True)
    one_rank = all(len(set(rk[pid[rows] == p].tolist())) == 1 for p in u[:50].tolist()) and \
        len(set(zip(pid[rows].tolist(), rk.tolist()))) == len(u)
    with np.load(K58.guard(K58.HEAD_NPZ), allow_pickle=False) as z:
        classes = set(np.asarray(z["class_id"]).astype(str).tolist())
    ik = set(E58.inner_keys_arrays()["player_id"].tolist())
    overlap = {"val_and_inner_split": len(val & inner), "val_and_inner_keys": len(val & ik),
               "val_and_s0_classes": len(val & classes), "inner_and_s0_classes": len(ik & classes)}
    okk = bool(set(u.tolist()) == val and one_rank and not np.asarray(cache.bad)[rows].any()
               and not any(overlap.values()) and cache.meta.get("games") == cache.n)
    if not okk:
        raise SystemExit(f"rows: validation players {len(u)} / {len(val)}, one rank {one_rank}, overlap {overlap}")
    os.makedirs(K.DATA, exist_ok=True)
    np.save(K.VAL_ROWS, rows)
    rep = {"format": "c10-prereg59 rows v1", "created": now(),
           "what": "every training game of the split v1 validation players (split == 'val' in the committed "
                   "results/split_v1/players_split.csv, as p49_lib.extra_players) = rows of the prereg49 move cache",
           "validation_players": int(len(u)), "validation_games": int(len(rows)),
           "games_per_player": {"min": int(n.min()), "median": float(np.median(n)), "max": int(n.max())},
           "players_per_rank": np.bincount(K58.rank_index(rk[first]), minlength=K.L).tolist(),
           "first_row": int(rows[0]), "last_row": int(rows[-1]), "overlap": overlap,
           "cache_codes_sha256": cache.meta.get("codes_sha256"), "cache_games": int(cache.n),
           "split_csv_sha256_lf": K58.sha_lf(K.SPLIT_CSV), "val_rows_sha256": K58.sha256_file(K.VAL_ROWS),
           "peak_working_set_gb": peak_gb(st)}
    K58.write_json(ROWS_JSON, rep)
    log(json.dumps({k: rep[k] for k in ("validation_players", "validation_games", "overlap", "peak_working_set_gb")}))
    return 0


# ================================================================ prototypes (PLAN A)
def feats_meta():
    with np.load(K58.guard(K58.FEATS_META), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in ("rank", "player_id")}


def build_prototypes(hrec, with_val):
    """D and E from S0's existing player-side store files (same store rows as prereg58's opponent-side files),
    validation players from embed_val.sh's file. -> (dict of arrays, record)."""
    meta = feats_meta()
    ids, P, rk, src, rec = [], [], [], [], {}
    for name, own_path, opp_path in (("D", K.OWN_D, K58.OPP_D), ("E", K.OWN_E, K58.OPP_E)):
        v = E58.load_vec(own_path)
        with np.load(K58.guard(opp_path), allow_pickle=False) as z:
            same_rows = bool(np.array_equal(z["store_row"], v["store_row"]))
        cfg = v["embed_cfg"]
        if not (same_rows and v["ckpt_sha256"] == hrec["ckpt_sha256"] and cfg["weights"] == "raw" and cfg["tta"] == 1):
            raise SystemExit(f"{own_path}: not S0 raw tta1 vectors of prereg58's store rows")
        pid = meta["player_id"][v["store_row"]].astype(str)
        prank = K58.rank_index(meta["rank"][v["store_row"]])
        u, pr, n = K.prototypes(v["emb"], v["count"], pid)
        first = {p: i for i, p in reversed(list(enumerate(pid.tolist())))}
        ids.append(u), P.append(pr), rk.append(prank[[first[p] for p in u.tolist()]]), src.append([name] * len(u))
        rec[name] = {"file": rel(own_path), "sha256": K58.sha256_file(own_path), "games": int(len(pid)),
                     "games_with_sample": int((v["count"] > 0).sum()), "players": int(len(set(pid.tolist()))),
                     "prototypes": int(len(u)), "store_rows_equal_prereg58_opponent_file": same_rows,
                     "games_per_prototype_min": int(n.min())}
    if with_val:
        _inner, val, L49 = split_players()
        v = E58.load_vec(K.OWN_VAL)
        rows = np.load(K.VAL_ROWS)
        cfg, rc = v["embed_cfg"], v["rows_cfg"]
        if not (np.array_equal(v["store_row"], rows) and v["ckpt_sha256"] == hrec["ckpt_sha256"]
                and cfg["weights"] == "raw" and cfg["tta"] == 1 and cfg["mode"] is None and rc["side"] == "own"
                and rc["cache_dir"] == "data/prereg49/cache" and np.isfinite(v["emb"]).all()):
            raise SystemExit(f"{K.OWN_VAL}: not the registered validation vectors (S0 raw tta1, own side, val_rows)")
        cache = L49.Cache49()
        pid = np.asarray(cache.pid)[rows].astype(str)
        prank = K58.rank_index(np.asarray(cache.rank)[rows])
        if set(pid.tolist()) != val:
            raise SystemExit("the validation vectors are not exactly the validation players' games")
        u, pr, n = K.prototypes(v["emb"], v["count"], pid)
        first = {p: i for i, p in reversed(list(enumerate(pid.tolist())))}
        ids.append(u), P.append(pr), rk.append(prank[[first[p] for p in u.tolist()]]), src.append(["V"] * len(u))
        rec["V"] = {"file": rel(K.OWN_VAL), "sha256": K58.sha256_file(K.OWN_VAL), "games": int(len(pid)),
                    "games_with_sample": int((v["count"] > 0).sum()), "players": int(len(val)),
                    "prototypes": int(len(u)), "games_per_prototype_min": int(n.min()),
                    "val_rows_sha256": K58.sha256_file(K.VAL_ROWS)}
    out = {"player_id": np.concatenate(ids), "proto": np.concatenate(P).astype(np.float64),
           "rank": np.concatenate(rk).astype(np.int64), "src": np.array(sum(src, []))}
    if len(set(out["player_id"].tolist())) != len(out["player_id"]):
        raise SystemExit("a player with two prototypes")
    rec["total"] = int(len(out["player_id"]))
    rec["per_rank"] = np.bincount(out["rank"], minlength=K.L).tolist()
    return out, rec


def save_prototypes(path, pr, hrec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".tmp.npz", player_id=pr["player_id"], proto=pr["proto"], rank=pr["rank"], src=pr["src"],
             ckpt_sha256=np.array(hrec["ckpt_sha256"]))
    os.replace(path + ".tmp.npz", path)
    return K58.sha256_file(path)


def load_prototypes(path):
    with np.load(K58.guard(path), allow_pickle=False) as z:
        return {"player_id": np.asarray(z["player_id"]).astype(str), "proto": np.asarray(z["proto"], np.float64),
                "rank": np.asarray(z["rank"], np.int64), "src": np.asarray(z["src"]).astype(str),
                "ckpt_sha256": str(z["ckpt_sha256"])}


# ================================================================ kernel (PLAN A, B)
def cmd_kernel(a):
    st = K58.limits(log)
    t0 = time.time()
    dry = bool(a.dry_de)
    if dry:
        out_dir = os.path.abspath(a.out_dir or "")
        if not a.out_dir or out_dir.lower().startswith(os.path.join(REPO, "results").lower()):
            raise SystemExit("--dry-de needs --out-dir outside results/")
        os.makedirs(out_dir, exist_ok=True)
        k_path, p_path = os.path.join(out_dir, "kernel.json"), os.path.join(out_dir, "prototypes.npz")
    else:
        k_path, p_path = K.KERNEL, K.PROTO
        if os.path.exists(K.KERNEL):
            raise SystemExit("kernel.json exists (chosen once)")
        if not committed_clean(ROWS_JSON):
            raise SystemExit("rows.json is not committed and clean")
    cs = code_state()
    if not dry and cs["prereg59_dirty"]:
        raise SystemExit(f"results/prereg59 not committed and clean: {cs['prereg59_dirty']}")
    g, Wn, wr, hrec, files = E58.kernel_games()          # opponent-side games of the D and E players
    pr, prec = build_prototypes(hrec, with_val=not dry)
    with np.load(K58.guard(K58.HEAD_NPZ), allow_pickle=False) as z:
        class_ids = set(np.asarray(z["class_id"]).astype(str).tolist())
    inner_pid = set(E58.inner_keys_arrays()["player_id"].tolist())
    players = sorted(set(g["player_id"].tolist()))
    proto_ids = set(pr["player_id"].tolist())
    overlap = {"kernel_and_inner_players": len(set(players) & inner_pid),
               "kernel_and_s0_classes": len(set(players) & class_ids),
               "prototypes_and_inner_players": len(proto_ids & inner_pid),
               "prototypes_and_s0_classes": len(proto_ids & class_ids)}
    if any(overlap.values()):
        raise SystemExit(f"player overlap: {overlap}")
    pindex = {p: i for i, p in enumerate(pr["player_id"].tolist())}
    own = np.array([pindex.get(p, -1) for p in g["player_id"].tolist()], np.int64)
    if (own < 0).any():
        raise SystemExit("a kernel player without a prototype")
    if not (pr["rank"][own] == K58.rank_index(g["rank"])).all():
        raise SystemExit("a kernel player's prototype has another rank")
    hd, jrec = E58.hd_join(g["rank"], g["game_id"], g["player_id"], g["color"])
    r_g = K58.rank_index(g["rank"])
    h_g, c_g = K58.hd_group(hd), K58.colour_index(g["color"])
    ok = g["count"] > 0
    pidx = {p: i for i, p in enumerate(players)}
    gp = np.array([pidx[p] for p in g["player_id"].tolist()], np.int64)
    p_rank = np.zeros(len(players), np.int64)
    p_rank[gp] = r_g
    p_src = np.empty(len(players), dtype="U1")
    p_src[gp] = g["src"]
    p_fold = np.array([K58.player_fold(p) for p in players], np.int64)
    f_g = p_fold[gp]
    p_sha = save_prototypes(p_path, pr, hrec)
    log(f"kernel games {len(gp)} ({int(ok.sum())} with a sample), players {len(players)}, classes "
        f"{len(wr)} head + {len(pr['rank'])} prototypes ({time.time() - t0:.0f}s)")
    parts = K.class_parts(g["emb"], Wn, wr, pr["proto"], pr["rank"], own)
    t_parts = time.time() - t0
    log(f"class parts done ({t_parts:.0f}s, peak {peak_gb(st)} GB)")
    grid = []
    best = None
    for t1, t2, b in K.GRID:
        ll, lln, by_fold = K.heldout_ll(K.combine(parts, t1, t2, b), r_g, h_g, c_g, f_g, ok)
        grid.append({"tau1": t1, "tau2": t2, "b": b, "heldout_mean_loglik": ll, "heldout_mean_loglik_by_fold": by_fold,
                     "record_only_heldout_mean_log_posterior_true_rank": lln})
        if best is None or ll > best[0]:
            best = (ll, t1, t2, b)
    _ll, t1, t2, b = best
    t_grid = time.time() - t0
    q = K.combine(parts, t1, t2, b)
    counts = K.table_counts(q, r_g, h_g, c_g, ok)
    M = K.table(counts)
    # record: the standalone competition score of the term on the kernel players, table cross-fitted by player fold
    rng = np.random.default_rng(SEED)
    SM = K58.score_matrix()
    Mf = K.fold_tables(q, r_g, h_g, c_g, f_g, ok)
    idx = np.arange(len(gp))
    by_p = [idx[ok & (gp == i)] for i in range(len(players))]
    res = {n: {"score": [], "rank": [], "src": []} for n in NS}
    prior_only = []
    for f in range(K.FOLDS):
        te = np.flatnonzero(ok & (f_g == f))
        lg = np.zeros((len(gp), K.L))
        lg[te] = K.game_ll(q[te], Mf[f], h_g[te], c_g[te])
        prior = np.bincount(p_rank[p_fold != f], minlength=K.L) + 1.0
        lprior = np.log(prior / prior.sum())
        for i in np.flatnonzero(p_fold == f).tolist():
            prior_only.append(float(SM[int(K58.decode(lprior[None])[0]), p_rank[i]]))
            for n in NS:
                if len(by_p[i]) < n:
                    continue
                draws = np.stack([lg[rng.choice(by_p[i], n, replace=False)].sum(0) for _ in range(DRAWS)])
                pred = K58.decode(lprior[None] + draws)
                res[n]["score"] += SM[pred, p_rank[i]].tolist()
                res[n]["rank"] += [int(p_rank[i])] * DRAWS
                res[n]["src"] += [str(p_src[i])] * DRAWS
    standalone = {"what": "the term alone (sum over the games of l_g + log prior; prior = rank frequencies of the other "
                          "folds' players), n random games of one kernel player, 20 draws per player, M from the other 4 "
                          "player folds at the chosen (tau1, tau2, b), own prototype left out; decoded by expected "
                          "competition score; record only", "prior_only_mean": float(np.mean(prior_only))}
    for n in NS:
        sc, rk, sr = np.array(res[n]["score"]), np.array(res[n]["rank"]), np.array(res[n]["src"])
        standalone[f"n{n}"] = {"mean": float(sc.mean()), "rank_balanced": K58.rank_balanced(sc, rk),
                               "questions": int(len(sc)), "D_players_mean": float(sc[sr == "D"].mean()),
                               "E_players_mean": float(sc[sr == "E"].mean()),
                               "by_rank": [float(sc[rk == k].mean()) for k in range(K.L)]}
    rep = {"format": "c10-prereg59 kernel v1", "dry_de": dry, "created": now(),
           "seconds": round(time.time() - t0, 1), "seconds_class_parts": round(t_parts, 1),
           "seconds_until_grid_done": round(t_grid, 1), "code": cs,
           "what": "PLAN.md A + B: classes = S0 head rows + prototypes; M[h, c][r][s] from soft counts q_g[s] at the "
                   "player's rank r with add-0.5 smoothing per row, on the D and E players of prereg36's split; "
                   "(tau1, tau2, b) by 5-fold (by player, prereg58's folds) held-out likelihood. No inner game is read.",
           "dry_note": ("dry run of the code path with the D / E prototypes only: NOT the registered kernel, not a "
                        "result") if dry else None,
           "grid": {"tau1": list(K.TAU1), "tau2": list(K.TAU2), "b": list(K.BIAS)},
           "tau1": int(t1), "tau2": int(t2), "b": int(b), "per_grid_point": grid,
           "selection_rule": "largest held-out mean log(sum_s q_g[s] M[h_g, c_g][r_g][s] + 1e-9) over the games of the "
                             "5 held-out player folds (NOTES.md 3); ties -> the first in tau1, tau2, b grid order",
           "index": {"h": ["HD 0", "HD 1", "HD 2+"], "c": ["player B", "player W"], "r": list(K.RANKS),
                     "s": list(K.RANKS)},
           "M": M.tolist(), "soft_counts": counts.tolist(), "smoothing": K.SMOOTH, "scale": K.SCALE,
           "games_per_h_c_r": [[[int((ok & (h_g == h) & (c_g == c) & (r_g == r)).sum()) for r in range(K.L)]
                                for c in range(K.N_COL)] for h in range(K.N_HD)],
           "classes": {"head": int(len(wr)), "prototypes": prec, "union": int(len(wr) + len(pr["rank"]))},
           "prototypes_file": {"file": rel(p_path) if not dry else "(scratch)", "sha256": p_sha},
           "players": {"total": len(players), "D": int((p_src == "D").sum()), "E": int((p_src == "E").sum()),
                       "per_rank": np.bincount(p_rank, minlength=K.L).tolist(),
                       "per_fold": np.bincount(p_fold, minlength=K.FOLDS).tolist()},
           "games": {"total": int(len(gp)), "with_sample": int(ok.sum()), "without_sample": int((~ok).sum()),
                     "D": int((g["src"] == "D").sum()), "E": int((g["src"] == "E").sum())},
           "overlap": overlap, "hd_join": jrec, "head": hrec, "vector_files": files,
           "standalone_score_record": standalone, "peak_working_set_gb": peak_gb(st)}
    K58.write_json(k_path, rep)
    if dry:
        log(json.dumps({"dry_de": True, "seconds": rep["seconds"], "seconds_class_parts": rep["seconds_class_parts"],
                        "seconds_until_grid_done": rep["seconds_until_grid_done"],
                        "peak_working_set_gb": rep["peak_working_set_gb"]}))
    else:
        log(json.dumps({"tau1": t1, "tau2": t2, "b": b, "classes": rep["classes"]["union"], "players": rep["players"],
                        "standalone": {k: v for k, v in standalone.items() if k != "what"},
                        "seconds": rep["seconds"], "peak_working_set_gb": rep["peak_working_set_gb"]}))
    return 0


def load_kernel(k_path, p_path, allow_dry=False):
    k = K58.read_json(k_path)
    if k.get("dry_de") and not allow_dry:
        raise SystemExit("kernel.json is a dry run")
    M = np.asarray(k["M"], np.float64)
    if M.shape != (K.N_HD, K.N_COL, K.L, K.L) or not np.allclose(M.sum(axis=3), 1.0):
        raise SystemExit("kernel.json: not a [3, 2, 10, 10] table")
    if K58.sha256_file(p_path) != k["prototypes_file"]["sha256"]:
        raise SystemExit("prototypes.npz is not the file of kernel.json")
    return M, (int(k["tau1"]), int(k["tau2"]), int(k["b"])), load_prototypes(p_path), k


# ================================================================ seam
def cmd_seam(a):
    st = K58.limits(log)
    if os.path.exists(K.SEAM):
        raise SystemExit("seam.json exists")
    _mods, _inner, _terms, _Emap, _q25, seam = E58.build_q25()
    rep = {"format": "c10-prereg59 seam v1", "created": now(), "code": code_state(),
           "what": "Q25 on inner through prereg58's code path (p58_eval.build_q25, imported unchanged); no number of "
                   "the new term is computed", **seam, "peak_working_set_gb": peak_gb(st)}
    K58.write_json(K.SEAM, rep)
    log(json.dumps({k: rep[k] for k in ("Q25", "Q25_mono", "Q25_bitwise", "Q25_mono_bitwise", "ok", "seconds",
                                        "peak_working_set_gb")}))
    return 0 if seam["ok"] else 1


# ================================================================ the judgement (once)
def term_values(emb, count, Wn, wr, pr, M, par, h, c):
    """l_g(r) [G, 10] of opponent-side vectors (no prototype left out) and the mean of max_s q_g[s]."""
    parts = K.class_parts(emb, Wn, wr, pr["proto"], pr["rank"], None, tau1=(par[0],), tau2=(par[1],))
    q = K.combine(parts, *par)
    lg = K.game_ll(q, M, h, c)
    lg[np.asarray(count) == 0] = 0.0
    return lg, float(q[np.asarray(count) > 0].max(axis=1).mean())


def inner_term(inner, M, par, pr, smoke):
    """Per-game l_g(r) on the full inner row space (0 for a game without an opponent-side sample or not in any
    question) and its record. smoke: the vectors are replaced by noise (no real value of the term on inner)."""
    Wn, wr, hrec = K58.load_head()
    v = E58.load_vec(K58.OPP_INNER)
    rows = np.load(K58.INNER_ROWS)
    if not (np.array_equal(v["store_row"], rows) and v["ckpt_sha256"] == hrec["ckpt_sha256"]
            and v["rows_cfg"]["side"] == "opp" and pr["ckpt_sha256"] == hrec["ckpt_sha256"]):
        raise SystemExit("inner opponent-side vectors / prototypes are not the registered files")
    keys = inner.d["inner"]["keys"]
    if set(pr["player_id"].tolist()) & set(np.asarray(keys["player_id"]).astype(str).tolist()):
        raise SystemExit("an inner player has a prototype")
    hd, jrec = E58.hd_join(keys["rank"], keys["game_id"], keys["player_id"], keys["color"])
    h, c = K58.hd_group(hd), K58.colour_index(keys["color"])
    emb = v["emb"]
    if smoke:      # nothing real of the term: noise vectors AND random HD group / colour (NOTES.md 14)
        rng = np.random.default_rng(NOISE_SEED)
        emb = rng.standard_normal(emb.shape).astype(np.float32)
        h, c = rng.integers(0, K.N_HD, len(h)), rng.integers(0, K.N_COL, len(c))
    vals, mq = term_values(emb, v["count"], Wn, wr, pr, M, par, h[rows], c[rows])
    lg = np.zeros((inner.n_inner, K.L))
    lg[rows] = vals
    if not np.isfinite(lg).all():
        raise SystemExit("the per-game term is not finite")
    rec = {"games_in_questions": int(len(rows)), "games_without_sample": int((v["count"] == 0).sum()),
           "classes": int(len(wr) + len(pr["rank"])), "hd_join": jrec,
           "hd_groups_of_question_games": np.bincount(h[rows], minlength=K.N_HD).tolist(),
           "player_colour_of_question_games": np.bincount(c[rows], minlength=K.N_COL).tolist(),
           "mean_max_q_record": mq}
    return lg, rec


def cmd_eval(a):
    st = K58.limits(log)
    t0 = time.time()
    smoke = bool(a.smoke)
    if smoke:
        if not a.kernel_dir:
            raise SystemExit("--smoke needs --kernel-dir (a kernel.json + prototypes.npz, e.g. of kernel --dry-de)")
        k_path, p_path = os.path.join(a.kernel_dir, "kernel.json"), os.path.join(a.kernel_dir, "prototypes.npz")
        out_path = os.path.join(SMOKE_DIR, "eval.json")
    else:
        k_path, p_path, out_path = K.KERNEL, K.PROTO, K.EVAL
        if a.kernel_dir:
            raise SystemExit("--kernel-dir is for --smoke only")
        if past_hard_stop():
            raise SystemExit(f"past the hard stop {HARD_STOP} (PLAN.md): no judgement")
        for p in (K.KERNEL, K.SEAM, ROWS_JSON):
            if not committed_clean(p):
                raise SystemExit(f"{os.path.basename(p)} is not committed and clean")
        if not K58.read_json(K.SEAM)["ok"]:
            raise SystemExit("seam.json is not ok")
        if os.path.exists(K.EVAL) or os.path.exists(K.STARTED):
            raise SystemExit("eval.json / eval_started.json exist: judged once")
    M, par, pr, krec = load_kernel(k_path, p_path, allow_dry=smoke)
    mods, inner, terms7, Emap, q25, seam = E58.build_q25()
    E41, E23, r, K23 = mods
    rf = r.rf
    cs = code_state()
    if not smoke and (cs["prereg59_dirty"] or cs["imported_dirty"]):
        raise SystemExit(f"not committed and clean: {cs['prereg59_dirty']} {cs['imported_dirty']}")
    log(f"seam {seam['ok']}: Q25 {q25['rb']!r}, single-colour {q25['mono_rb']!r} ({time.time() - t0:.0f}s)")
    if not seam["ok"]:
        raise SystemExit("seam failed inside the judgement run: no number of the new term is computed")
    if not smoke:
        K58.write_json(K.STARTED, {"format": "c10-prereg59 eval started", "created": r.now_iso(), "code": cs,
                                   "tau1_tau2_b": list(par), "kernel_sha256_lf": K58.sha_lf(K.KERNEL)})
    # ---- the new term (the first inner number of it is computed below)
    lg, trec = inner_term(inner, M, par, pr, smoke)
    if smoke:
        trec = {"smoke": "inner opponent vectors, HD groups and colours replaced by default_rng(12345) draws", **trec}
    di = inner.d["inner"]
    E59 = {w: K.question_term(S, lg) for w, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]),
                                                   ("mono", di["S_mono"]))}
    terms8 = terms7 + [TERM]
    cand = inner.run(terms8, {**Emap, TERM: E59})
    cmp_ = inner.compare(cand, q25)
    gates = K23.gates(cmp_["inner_diff"], cmp_["inner_ci95"], cmp_["mono_diff"], cmp_["mono_ci95"],
                      cmp_["mono_by_colour"])
    adopt = bool(gates["passed"]) and not smoke
    log(f"Q59 {cand['rb']!r} vs Q25 {q25['rb']!r}: {json.dumps(gates)}")
    # ---- record only
    ones = {w: np.ones(len(E59[w]), bool) for w in ("eval", "mono")}
    alone_scaled = E23.alone(inner, {w: E59[w] for w in ("eval", "mono")}, ones)
    alone_sum = E23.alone(inner, {"eval": E59["eval"] / K.SCALE * inner.Xe.n[:, None],
                                  "mono": E59["mono"] / K.SCALE * inner.Xm.n[:, None]}, ones)
    w_new = [float(fr["params"]["w"][TERM]) for fr in cand["folds"]]
    rep = {"format": "c10-prereg59 eval v1", "smoke": smoke, "created": r.now_iso(),
           "seconds": round(time.time() - t0, 1), "code": cs,
           "smoke_note": "inner opponent vectors replaced by noise: numbers meaningless" if smoke else None,
           "what": "PLAN.md: inner-only judgement, once, of Q59 = Q25's seven terms + OPP2 (10 x the pooled mean over "
                   "the question's games of l_g(r) = log(sum_s q_g[s] M[h_g, c_g][r][s] + 1e-9)), one more E term of "
                   "Inner.run, 5-fold player OOF fusion, rule B; no validation scoring, no test data. Designed after "
                   "prereg58's judgement on the same inner questions (inner reused).",
           "adopt": adopt,
           "status": ("all three gates passed: freeze the 8-term fusion, write PREREG_59 (PLAN.md)" if adopt
                      else "gates not passed: not adopted"),
           "tau1": par[0], "tau2": par[1], "b": par[2], "scale": K.SCALE,
           "kernel_sha256_lf": K58.sha_lf(k_path), "prototypes_sha256": krec["prototypes_file"]["sha256"],
           "seam": seam, "term": trec,
           "candidate": {"name": "Q59", "terms": terms8, **inner.short(cand, q25)},
           "baseline": {"name": "Q25", "terms": terms7, "inner_rank_balanced": q25["rb"],
                        "single_colour_rank_balanced": q25["mono_rb"],
                        "single_colour_by_colour": {c: q25["mrec"]["by_colour"][c]["rank_balanced"]
                                                    for c in q25["mrec"]["by_colour"]}},
           "gates": gates,
           "weight_of_new_term": {"by_fold": w_new, "limit": K.W_BOUND,
                                  "at_limit_by_fold": [bool(w >= K.W_BOUND - 1e-8) for w in w_new],
                                  "at_zero_by_fold": [bool(w <= 1e-8) for w in w_new]},
           "record_only": {"OPP2_alone_rule_b_scaled_mean": alone_scaled,
                           "OPP2_alone_rule_b_sum_over_games": alone_sum,
                           "by_game_count": E23.by_count(inner, cand, q25),
                           "by_level": E23.by_level(inner, cand, q25),
                           "weights": E23.w_record(cand, terms8),
                           "q25_weights": E23.w_record(q25, terms7),
                           "fusion_a_gamma_by_fold": [{"fold": fr["fold"], "a": fr["params"]["a"],
                                                       "gamma": fr["params"]["gamma"]} for fr in cand["folds"]]},
           "peak_working_set_gb": peak_gb(st)}
    K58.write_json(out_path, rep)
    log(f"-> {out_path}: {rep['status']} ({time.time() - t0:.0f}s, peak {peak_gb(st)} GB)")
    return 0


# ================================================================ RESULTS.md (mechanical)
def f4(x):
    return "n/a" if x is None else f"{x:+.4f}"


def results_text(e, k):
    c, b, g, ro = e["candidate"], e["baseline"], e["gates"], e["record_only"]
    vs = c["vs"]
    sa = k["standalone_score_record"]
    w = e["weight_of_new_term"]
    L_ = []
    A = L_.append
    A("# prereg59 (competition 2): opponent-identity term, second version - " + ("ADOPT (gates passed)" if e["adopt"]
                                                                                 else "NOT adopted"))
    A("")
    A(f"Written mechanically by `p59_eval.py results` from `eval.json` ({e['created']}) and `kernel.json` "
      f"({k['created']}); numbers only. Plan `PLAN.md`, narrow readings `NOTES.md`. This version was designed after "
      "prereg58's judgement on the same inner questions (inner reused).")
    if e.get("smoke") or k.get("dry_de"):
        A("")
        A("**SMOKE / DRY FILES: the numbers below are meaningless (pipeline check only).**")
    A("")
    A("## Kernel")
    A("")
    pc = k["classes"]["prototypes"]
    A(f"- Classes: {k['classes']['head']} head rows + {pc['total']} prototypes ("
      + ", ".join(f"{n} {pc[n]['prototypes']}" for n in ("D", "E", "V") if n in pc) + f") = {k['classes']['union']}.")
    A(f"- Kernel players {k['players']['total']} (D {k['players']['D']}, E {k['players']['E']}), games "
      f"{k['games']['total']} ({k['games']['without_sample']} without a sample).")
    best = max(k["per_grid_point"], key=lambda x: x["heldout_mean_loglik"])
    A(f"- Chosen tau1 = {k['tau1']}, tau2 = {k['tau2']}, b = {k['b']} (held-out mean log-likelihood "
      f"{best['heldout_mean_loglik']:.4f}; {len(k['per_grid_point'])} grid points, range "
      f"{min(x['heldout_mean_loglik'] for x in k['per_grid_point']):.4f} .. {best['heldout_mean_loglik']:.4f}).")
    A("- Standalone record on the kernel players (5 / 10 / 20 games): "
      + " / ".join(f"{sa['n%d' % n]['mean']:.4f}" for n in (5, 10, 20))
      + f" (prior only {sa['prior_only_mean']:.4f}).")
    A("")
    A("## Seam")
    A("")
    A(f"Q25 inner {e['seam']['Q25'][0]!r} (reference {e['seam']['Q25'][1]!r}), single colour "
      f"{e['seam']['Q25_mono'][0]!r}; bit for bit: {e['seam']['Q25_bitwise']} / {e['seam']['Q25_mono_bitwise']}.")
    A("")
    A("## Judgement (inner, once)")
    A("")
    A("| | Q25 | Q59 | diff | 95% interval | gate |")
    A("|---|---|---|---|---|---|")
    A(f"| inner eval | {b['inner_rank_balanced']:.7f} | {c['inner_rank_balanced']:.7f} | {vs['inner_diff']:+.5f} | "
      f"[{vs['inner_ci95'][0]:+.5f}, {vs['inner_ci95'][1]:+.5f}] | G1 {'pass' if g['G1']['passed'] else 'no'}; "
      f"G2 {'pass' if g['G2']['passed'] else 'no'} |")
    A(f"| single colour | {b['single_colour_rank_balanced']:.7f} | {c['single_colour_rank_balanced']:.7f} | "
      f"{vs['mono_diff']:+.5f} | [{vs['mono_ci95'][0]:+.5f}, {vs['mono_ci95'][1]:+.5f}] | "
      f"G4 {'pass' if g['G4']['passed'] else 'no'} (B {f4(vs['mono_by_colour'].get('B'))}, "
      f"W {f4(vs['mono_by_colour'].get('W'))}) |")
    A("")
    A(f"**adopt = {str(e['adopt']).lower()}** ({e['status']}).")
    A("")
    A("## Record only")
    A("")
    A("- Weight of the new term by fold: " + ", ".join(f"{x:.4f}" for x in w["by_fold"])
      + f" (limit {w['limit']}; at the limit in {sum(w['at_limit_by_fold'])} of {len(w['by_fold'])} folds, at zero in "
      f"{sum(w['at_zero_by_fold'])}).")
    A(f"- The term alone (rule B): scaled mean {ro['OPP2_alone_rule_b_scaled_mean']['inner_rank_balanced']:.4f} "
      f"(single colour {ro['OPP2_alone_rule_b_scaled_mean']['single_colour_rank_balanced']:.4f}); sum over games "
      f"{ro['OPP2_alone_rule_b_sum_over_games']['inner_rank_balanced']:.4f} "
      f"({ro['OPP2_alone_rule_b_sum_over_games']['single_colour_rank_balanced']:.4f}).")
    A("- By rank class (eval, Q59 - Q25): " + ", ".join(f"{x['level']} {x['diff']:+.4f}" for x in ro["by_level"]["eval"])
      + ".")
    A("- By rank class (single colour): " + ", ".join(f"{x['level']} {x['diff']:+.4f}" for x in ro["by_level"]["mono"])
      + ".")
    bc = ro["by_game_count"]
    A("- Single colour by number of games: " + "; ".join(
        f"{n} games {bc['mono_by_k'][n]['diff']:+.4f} [{bc['mono_by_k'][n]['ci95'][0]:+.4f}, "
        f"{bc['mono_by_k'][n]['ci95'][1]:+.4f}]" for n in sorted(bc["mono_by_k"], key=int)) + ".")
    A("- Eval questions with exactly n games: " + "; ".join(
        f"{n} games {bc['eval_by_n_exact'][n]['diff']:+.4f} [{bc['eval_by_n_exact'][n]['ci95'][0]:+.4f}, "
        f"{bc['eval_by_n_exact'][n]['ci95'][1]:+.4f}]" for n in sorted(bc["eval_by_n_exact"], key=int)) + ".")
    A(f"- Inner question games {e['term'].get('games_in_questions')}, without a sample "
      f"{e['term'].get('games_without_sample')}; mean of max_s q_g[s] {e['term'].get('mean_max_q_record'):.4f}.")
    A("")
    return "\n".join(L_)


def cmd_results(a):
    if a.from_dir:
        if not a.out or os.path.abspath(a.out).lower().startswith(os.path.join(REPO, "results").lower()):
            raise SystemExit("--from-dir needs --out outside results/")
        e_path, k_path, out = os.path.join(a.from_dir, "eval.json"), os.path.join(a.from_dir, "kernel.json"), a.out
    else:
        e_path, k_path, out = K.EVAL, K.KERNEL, K.RESULTS
        for p in (K.EVAL, K.KERNEL):
            if not committed_clean(p):
                raise SystemExit(f"{os.path.basename(p)} is not committed and clean")
        if os.path.exists(out):
            raise SystemExit("RESULTS.md exists")
    text = results_text(K58.read_json(e_path), K58.read_json(k_path))
    with open(K58.guard(out), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    log(f"-> {out}")
    return 0


def cmd_summary(a):
    """One line for the unattended chain's commit messages / checks (reads the committed json only). seam: exit 1
    when it is not ok."""
    if a.what == "kernel":
        k = K58.read_json(K.KERNEL)
        sa = k["standalone_score_record"]
        print(f"tau1 {k['tau1']}, tau2 {k['tau2']}, b {k['b']}; {k['classes']['union']} classes "
              f"({k['classes']['head']} head + {k['classes']['prototypes']['total']} prototypes); standalone record "
              + " / ".join(f"{sa['n%d' % n]['mean']:.3f}" for n in NS) + " (n = 5 / 10 / 20)")
        return 0
    if a.what == "seam":
        s = K58.read_json(K.SEAM)
        print(f"Q25 {s['Q25'][0]!r}, single colour {s['Q25_mono'][0]!r}, bit for bit {s['Q25_bitwise']} / "
              f"{s['Q25_mono_bitwise']}")
        return 0 if s["ok"] else 1
    e = K58.read_json(K.EVAL)
    vs, g = e["candidate"]["vs"], e["gates"]
    print(f"Q59 {e['candidate']['inner_rank_balanced']:.7f} vs Q25 {e['baseline']['inner_rank_balanced']:.7f} "
          f"({vs['inner_diff']:+.4f}, CI [{vs['inner_ci95'][0]:+.4f}, {vs['inner_ci95'][1]:+.4f}]); single colour "
          f"{vs['mono_diff']:+.4f} [{vs['mono_ci95'][0]:+.4f}, {vs['mono_ci95'][1]:+.4f}]; G1 "
          f"{'pass' if g['G1']['passed'] else 'no'}, G2 {'pass' if g['G2']['passed'] else 'no'}, G4 "
          f"{'pass' if g['G4']['passed'] else 'no'} - " + ("ADOPT (gates passed)" if e["adopt"] else "not adopted"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rows")
    p = sub.add_parser("kernel")
    p.add_argument("--dry-de", action="store_true")
    p.add_argument("--out-dir", default=None)
    sub.add_parser("seam")
    p = sub.add_parser("eval")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--kernel-dir", default=None)
    p = sub.add_parser("results")
    p.add_argument("--from-dir", default=None)
    p.add_argument("--out", default=None)
    p = sub.add_parser("summary")
    p.add_argument("what", choices=["kernel", "seam", "eval"])
    a = ap.parse_args(argv)
    return {"rows": cmd_rows, "kernel": cmd_kernel, "seam": cmd_seam, "eval": cmd_eval,
            "results": cmd_results, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
