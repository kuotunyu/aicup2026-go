"""prereg45 explore: seam, D selection, the one E judgement, official pools (PLAN.md 2-4). CPU only.

    PY results/prereg45/explore/p45_eval.py seam                         # -> seam.json (v2-6 rebuilt, D and E)
    PY results/prereg45/explore/p45_eval.py select --sets L,M1,M2        # -> tune.json (commit before judge)
    PY results/prereg45/explore/p45_eval.py select --sets L,M1           # PLAN.md 1 fall-back (M2 unfinished at
                                                                         #   11/15 22:00 only)
    PY results/prereg45/explore/p45_eval.py judge --e-use 6              # once -> eval.json
    PY results/prereg45/explore/p45_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg45/explore/p45_eval.py assemble --src <official tests dir>

Base = v2-6 = the prereg43 system as recorded and adopted (results/prereg43/explore/tune.json + eval.json): c* (S0,
raw) vectors, prereg38 backend cfg of c*, the KataGo P1 term joint / b4 with the LDA64 learned on L (prereg39),
wP 8, CSLS 20. A system here = v2-6 with only the P1 projection replaced (PLAN.md 2):
    S = CSLS_k( fuse_cfg(pool, e_term(pool, emb_c*, cfg_c*), cfg_c*) + wP z(b4 term of the projected P1 games) ).
Variants (fixed order): A_LDA64, A_LDA128 (the LDA of tools/p10_fuse.fit_p1_lda, learned on S = L + M1 [+ M2],
computed in row blocks so that one process stays < 2 GB; same formulas), B_MLP (standardise -> 512 -> GELU -> 128 ->
L2, cosine softmax scale 30 over the S players; AdamW 1e-3 / 1e-4, batch 1,024, 20 epochs, seed 0, CPU torch,
deterministic algorithms, 4 threads). Each variant: (wP, k) on D over prereg43's grid (first maximum); c45 = the
first variant with the highest D mean. Gate: D(c45) - 0.7467766213119217 >= 0.01.
judge: c45 vs v2-6 on the 20 E pools, once (E pool use 6): mean >= +0.01, pool bootstrap (10,000, seed 43) 95% low
> 0, higher in >= 18 of 20. Nothing here opens the validation-answer folder or runs tools/score.py.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P43 = os.path.join(REPO, "results", "prereg43", "explore")
for p in (P43, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p43_eval as E43  # noqa: E402   (v2-6's code path: fused, scores, p1_ctx, p1_z, p1_grid, judge_rule, ...)

P39E, Q42 = E43.P39E, E43.Q42
BK, EV, st, sim, L = E43.BK, E43.EV, E43.st, E43.sim, E43.L
F10 = P39E.F10

DATA = os.path.join(REPO, "data", "prereg45")
LISTS = os.path.join(DATA, "lists")
P1_45 = os.path.join(DATA, "p1")
MODELS = os.path.join(DATA, "models")
RECORD = os.path.join(HERE, "lists_record.json")
V26_TUNE = os.path.join(P43, "tune.json")
V26_EVAL = os.path.join(P43, "eval.json")
V26_APPLY = os.path.join(REPO, "results", "prereg43", "apply")
OUT_APPLY = os.path.join(REPO, "results", "prereg45", "apply")
V1_SUBMISSION = os.path.join(REPO, "results", "test06", "player", "on", "submission_player.csv")
D_V26 = 0.7467766213119217
E_V26 = 0.681614485210018
VARIANTS = ("A_LDA64", "A_LDA128", "B_MLP")
LDA_DIMS = {"A_LDA64": 64, "A_LDA128": 128}
SET_CHOICES = {"L,M1,M2": ("L", "M1", "M2"), "L,M1": ("L", "M1")}
FALLBACK_AT = datetime.datetime(2026, 11, 15, 22, 0)
REV1_M2_DROPPED = True   # PLAN.md revision 1 (11/15 08:40, before any score): M2 is not extracted; L + M1 once M1 is complete        # PLAN.md 1 (Taiwan time = this machine's time)
WP_GRID, K_GRID = E43.WP_GRID, E43.K_GRID
P1_VARIANT, P1_PW = E43.P1_VARIANT, E43.P1_PW
GATE = 0.01
COMP1_E_JUDGES = E43.COMP1_E_JUDGES + ("prereg43", "prereg45")
MLP = {"hidden": 512, "out": 128, "scale": 30.0, "lr": 1e-3, "wd": 1e-4, "batch": 1024, "epochs": 20, "seed": 0,
       "threads": 4}
BLOCK = 20000
RAM_NEED_MB = 9000


# ================================================================ small helpers
_abs, _rel, read_json, first_max = E43._abs, E43._rel, E43.read_json, E43.first_max


def peak_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 1048576.0, 1)
    except Exception:          # noqa: BLE001
        return None


def arrays_sha(arrs, keys):
    h = hashlib.sha256()
    for k in keys:
        a = np.ascontiguousarray(arrs[k])
        h.update(k.encode() + b"|" + str(a.dtype).encode() + b"|" + str(a.shape).encode() + b"|")
        h.update(a.tobytes())
    return h.hexdigest()


# ================================================================ the base (v2-6)
def v26():
    """v2-6 exactly as prereg43 recorded and adopted it."""
    t, ev = read_json(V26_TUNE), read_json(V26_EVAL)
    if not ev.get("adopt") or ev.get("tune_sha256") != st.sha256_lf(V26_TUNE):
        raise SystemExit("prereg43 eval.json does not adopt prereg43 tune.json")
    cs = t["c_star"]
    if (cs["p1"]["variant"], cs["p1"]["pw"]) != (P1_VARIANT, P1_PW):
        raise SystemExit("v2-6's P1 structure is not joint / b4")
    if cs["p1"]["wP"] not in WP_GRID or cs["p1"]["csls_k"] not in K_GRID:
        raise SystemExit("v2-6's (wP, k) is not in the grid")
    if cs["p1"]["mean"] != D_V26 or ev["means"]["c_star"] != E_V26:
        raise SystemExit("prereg43 records do not hold v2-6's D / E means")
    return {"tune": t, "eval": ev, "cfg": cs["backend"]["cfg"], "wP": cs["p1"]["wP"], "k": cs["p1"]["csls_k"],
            "lda_sha256": t["lda"]["sha256"], "d_per_pool": list(cs["p1"]["per_pool"]),
            "d_backend_per_pool": list(cs["backend"]["per_pool"]), "e_per_pool": list(ev["per_pool"]["c_star"]),
            "e_backend_per_pool": list(ev["per_pool"]["c_star_backend_only"]),
            "record": {"tune": _rel(V26_TUNE), "eval": _rel(V26_EVAL), "tune_sha256_lf": st.sha256_lf(V26_TUNE),
                       "eval_sha256_lf": st.sha256_lf(V26_EVAL), "c_star": {x: cs[x] for x in
                                                                             ("label", "members", "ckpt_sha256")},
                       "cfg": cs["backend"]["cfg"], "p1": {x: cs["p1"][x] for x in ("variant", "pw", "wP",
                                                                                       "csls_k", "mean")},
                       "lda_sha256": t["lda"]["sha256"], "d_mean": D_V26, "e_mean": E_V26}}


def base_fused(base, which, pools):
    """v2-6's CNN + backend part (shared by every prereg45 variant) on the D / E pools, with its backend-only check."""
    path, _choice = E43.cstar_file(base["tune"], which)
    E43.check_players(path, which)
    e = E43.load_emb(path)
    Fs = E43.fused(pools, e, base["cfg"])
    del e
    kb = int(base["cfg"]["norm"][1])
    backend = [float(st.score(st.csls(F, kb), p["owner"])) for F, p in zip(Fs, pools)]
    want = base["d_backend_per_pool"] if which == "D" else base["e_backend_per_pool"]
    if backend != want:
        raise SystemExit("seam %s: v2-6's backend-only per-pool scores differ from prereg43's records" % which)
    return Fs, {"file": _rel(path), "file_sha256": L.sha256_file(path)}


def seam_scores(base, which, Fs, zs, pools):
    per = E43.scores(Fs, zs, base["wP"], base["k"], pools)
    want, mean = (base["d_per_pool"], D_V26) if which == "D" else (base["e_per_pool"], E_V26)
    if per != want or float(np.mean(per)) != mean:
        raise SystemExit("seam %s: v2-6 rebuilt = %r, recorded %r" % (which, float(np.mean(per)), mean))
    return per


# ================================================================ the learning set S (PLAN.md 1, 2)
def read_list45(name):
    p = os.path.join(LISTS, name + ".json")
    want = read_json(RECORD)["lists_sha256"][name]
    if L.sha256_file(p) != want:
        raise SystemExit("%s differs from prereg45 lists_record.json" % _rel(p))
    return read_json(p)["games"]


def set_source(name):
    """-> (games, chunk dir). L = prereg39's list and chunks; M1 / M2 = prereg45's."""
    if name == "L":
        return P39E.read_list("L"), P39E.P1
    return read_list45(name), P1_45


def chunks_complete(name, games=None, p1_dir=None):
    """Every chunk of the list is written (no .tmp counts): ceil(n / 200) chunk files for the fresh games."""
    if games is None:
        games, p1_dir = set_source(name)
    n = sum(1 for g in games if g.get("mock_ref", -1) < 0)
    d = os.path.join(p1_dir, name)
    have = len(F10.chunk_files(d)) if os.path.isdir(d) else 0
    return have == (n + P39E.CHUNK - 1) // P39E.CHUNK, have, (n + P39E.CHUNK - 1) // P39E.CHUNK


def check_sets_choice(choice, complete, now):
    """PLAN.md 1: L + M1 + M2 when M2 is complete; L + M1 only if M2 is not complete at / after 11/15 22:00;
    M1 incomplete -> stop. Scores are never looked at here. complete: {name: bool}."""
    names = SET_CHOICES.get(choice)
    if names is None:
        raise SystemExit("--sets must be one of %s" % sorted(SET_CHOICES))
    if not complete.get("M1"):
        raise SystemExit("M1 is not fully extracted: prereg45 stops without a judgement (PLAN.md 1)")
    if names == ("L", "M1"):
        if complete.get("M2"):
            raise SystemExit("M2 is complete: use --sets L,M1,M2 (the fall-back is for an unfinished M2 only)")
        if now < FALLBACK_AT and not REV1_M2_DROPPED:
            raise SystemExit("the L,M1 fall-back starts at %s (PLAN.md 1); M2 is still being extracted" % FALLBACK_AT)
    elif not complete.get("M2"):
        raise SystemExit("M2 is not fully extracted: wait, or after %s use --sets L,M1" % FALLBACK_AT)
    return names


def load_S(names):
    """S = the named lists' P1 games in one float32 array (one copy; each set is freed after it is copied).
    -> dict(X, usable, labels, colour, sets (per-set record), meta)."""
    srcs = [(n,) + set_source(n) for n in names]
    for n, games, d in srcs:
        ok, have, want = chunks_complete(n, games, d)
        if not ok:
            raise SystemExit("%s: %d of %d chunks: not complete" % (n, have, want))
    total = sum(len(g) for _n, g, _d in srcs)
    X = np.zeros((total, F10.P1_DIM), np.float32)
    usable = np.zeros(total, bool)
    labels, colour, rec, meta = [], [], {}, None
    o = 0
    for n, games, d in srcs:
        s = P39E.load_set(n, p1_dir=d, games=games)
        k = len(games)
        X[o:o + k] = s["X"]
        usable[o:o + k] = s["usable"]
        labels += [g["player_id"] for g in games]
        colour += [g["color"] for g in games]
        if meta is None:
            meta = s["meta"]
        elif s["meta"] != meta:
            raise SystemExit("%s chunks were made with other code / weights than L's" % n)
        if not np.isfinite(s["X"][s["usable"]]).all():
            raise SystemExit("%s: non-finite P1 values (PLAN.md revision 1)" % n)
        if n != "L" and int(np.sum(s["failed"])) != 0:
            raise SystemExit("%s: failed games in the chunks (PLAN.md revision 1)" % n)
        rec[n] = {"games": k, "usable": int(s["usable"].sum()), "failed_fresh": s["failed"],
                  "players": len({g["player_id"] for g in games}),
                  "list_sha256": L.sha256_file(os.path.join(P39E.LISTS if n == "L" else LISTS, n + ".json"))}
        o += k
        del s
    keys = [(g["rank"], g["game_id"], g["player_id"]) for _n, gl, _d in srcs for g in gl]
    if len(keys) != len(set(keys)):
        raise SystemExit("a game is in two of the S lists")
    E, D, T, _per = E43.splits()
    if not set(labels) <= set(T):
        raise SystemExit("an S player is not a T (training) player")
    return {"X": X, "usable": usable, "labels": np.asarray(labels), "colour": np.asarray(colour), "sets": rec,
            "meta": meta}


# ================================================================ variant A: LDA (p10_fuse.fit_p1_lda, in blocks)
def _keep_rows(labels, mask, min_games=F10.MIN_GAMES):
    """fit_p1_lda's row selection: rows in mask whose player has >= min_games such rows."""
    idx = np.flatnonzero(np.asarray(mask, bool))
    lab = np.asarray(labels)[idx]
    _u, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= min_games
    idx, lab = idx[keep], lab[keep]
    if len(set(lab.tolist())) < 2:
        raise SystemExit("fewer than 2 players with enough usable games")
    return idx, lab


def _col_mean(X, idx, f=None, block=BLOCK):
    acc = np.zeros(X.shape[1])
    for s in range(0, len(idx), block):
        B = np.asarray(X[idx[s:s + block]], np.float64)
        acc += (f(B) if f else B).sum(0)
    return acc / len(idx)


def fit_lda_blocked(X, labels, mask, d, block=BLOCK):
    """tools/p10_fuse.fit_p1_lda + p06_style.lda_fit (reg LDA_REG, eigh(Sb, Sw_r), first d directions) with every
    pass over the rows done in blocks of `block` rows (float64 inside a block), so S (~173k x 770) never exists in
    float64. Same formulas; the values agree with fit_p1_lda to rounding (test_p45, seam.json)."""
    idx, lab = _keep_rows(labels, mask)
    m = _col_mean(X, idx, block=block)
    s = np.sqrt(_col_mean(X, idx, lambda B: (B - m) ** 2, block)) + F10.STD_EPS
    mu = _col_mean(X, idx, lambda B: (B - m) / s, block)
    sd = np.sqrt(_col_mean(X, idx, lambda B: ((B - m) / s - mu) ** 2, block))
    sd[sd < 1e-9] = 1.0
    _uniq, inv = np.unique(lab, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    D = X.shape[1]
    M = np.zeros((len(n), D))
    for a in range(0, len(idx), block):
        Z = ((np.asarray(X[idx[a:a + block]], np.float64) - m) / s - mu) / sd
        np.add.at(M, inv[a:a + block], Z)
    M /= n[:, None]
    Sw = np.zeros((D, D))
    for a in range(0, len(idx), block):
        Z = ((np.asarray(X[idx[a:a + block]], np.float64) - m) / s - mu) / sd - M[inv[a:a + block]]
        Sw += Z.T @ Z
    Sw /= len(idx)
    Sb = (M * n[:, None]).T @ M / len(idx)
    Sw_r = Sw + st.LDA_REG * np.trace(Sw) / D * np.eye(D)
    ev, W = sla.eigh(Sb, Sw_r)
    order = np.argsort(-ev)
    return {"kind": "lda", "m": m, "s": s, "mu": mu, "sd": sd, "W": W[:, order][:, :d], "ev": ev[order][:d],
            "n_games": int(len(idx)), "n_players": int(len(n))}


LDA_KEYS = ("m", "s", "mu", "sd", "W", "ev")


def project_lda(X, model, block=BLOCK):
    """F10.project_p1 with the model's own dimension: ((X - m) / s - mu) / sd @ W, in row blocks."""
    out = np.zeros((len(X), model["W"].shape[1]))
    for a in range(0, len(X), block):
        B = np.asarray(X[a:a + block], np.float64)
        out[a:a + len(B)] = (((B - model["m"]) / model["s"] - model["mu"]) / model["sd"]) @ model["W"]
    return out


# ================================================================ variant B: MLP (cosine softmax)
MLP_KEYS = ("m", "s", "w1", "b1", "w2", "b2")


def _torch():
    import torch
    torch.set_num_threads(MLP["threads"])
    torch.use_deterministic_algorithms(True)
    return torch


def _standardise_inplace(X, m, s, block=BLOCK):
    for a in range(0, len(X), block):
        X[a:a + block] = ((np.asarray(X[a:a + block], np.float64) - m) / s).astype(np.float32)


def train_mlp(X, labels, mask, cfg=MLP, log=print, consume=False):
    """PLAN.md 2 B_MLP. Standardisation = fit_p1_lda's (mean, sd + 1e-6 of the kept rows). consume=True
    standardises X in place (the caller no longer needs the raw values; saves one S-sized copy)."""
    torch = _torch()
    t0 = time.time()
    idx, lab = _keep_rows(labels, mask)
    m = _col_mean(X, idx)
    s = np.sqrt(_col_mean(X, idx, lambda B: (B - m) ** 2)) + F10.STD_EPS
    if not consume:
        X = np.array(X, np.float32, copy=True)
    _standardise_inplace(X, m, s)
    _u, y = np.unique(lab, return_inverse=True)
    n_cls = int(y.max()) + 1
    torch.manual_seed(cfg["seed"])
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], cfg["hidden"]), torch.nn.GELU(),
                              torch.nn.Linear(cfg["hidden"], cfg["out"]))
    head = torch.nn.Parameter(torch.randn(n_cls, cfg["out"]) * 0.01)
    opt = torch.optim.AdamW(list(net.parameters()) + [head], lr=cfg["lr"], weight_decay=cfg["wd"])
    Xt = torch.from_numpy(X)
    rows = torch.from_numpy(idx.astype(np.int64))
    yt = torch.from_numpy(y.astype(np.int64))
    gen = torch.Generator().manual_seed(cfg["seed"])
    hist = []
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(len(rows), generator=gen)
        tot, cor, nb = 0.0, 0, 0
        for a in range(0, len(perm), cfg["batch"]):
            b = perm[a:a + cfg["batch"]]
            h = torch.nn.functional.normalize(net(Xt[rows[b]]), dim=1)
            logits = cfg["scale"] * h @ torch.nn.functional.normalize(head, dim=1).T
            loss = torch.nn.functional.cross_entropy(logits, yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
            cor += int((logits.argmax(1) == yt[b]).sum())
            nb += len(b)
        hist.append({"epoch": ep + 1, "loss": tot / nb, "train_acc": cor / nb, "seconds": round(time.time() - t0, 1)})
        log(json.dumps({"B_MLP": hist[-1]}))
    sd = net.state_dict()
    model = {"kind": "mlp", "m": m, "s": s, "w1": sd["0.weight"].numpy().copy(), "b1": sd["0.bias"].numpy().copy(),
             "w2": sd["2.weight"].numpy().copy(), "b2": sd["2.bias"].numpy().copy(),
             "n_games": int(len(idx)), "n_players": n_cls}
    return model, {"history": hist, "seconds": round(time.time() - t0, 1), "torch": torch.__version__,
                   "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled()}


def project_mlp(X, model, block=BLOCK):
    """Standardise (float64 -> float32), the trained net, L2 -> float64 [n, 128]."""
    torch = _torch()
    W1, b1 = torch.from_numpy(model["w1"]), torch.from_numpy(model["b1"])
    W2, b2 = torch.from_numpy(model["w2"]), torch.from_numpy(model["b2"])
    out = np.zeros((len(X), W2.shape[0]))
    with torch.no_grad():
        for a in range(0, len(X), block):
            B = torch.from_numpy(((np.asarray(X[a:a + block], np.float64) - model["m"]) / model["s"])
                                 .astype(np.float32))
            h = torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(B, W1, b1)), W2, b2)
            out[a:a + len(B)] = torch.nn.functional.normalize(h, dim=1).numpy().astype(np.float64)
    return out


# ================================================================ models on disk
def model_keys(model):
    return LDA_KEYS if model["kind"] == "lda" else MLP_KEYS


def model_sha(model):
    return arrays_sha(model, model_keys(model))


def save_model(variant, model, d=MODELS):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, variant + ".npz")
    tmp = p + ".tmp.npz"
    np.savez(tmp, kind=np.array(model["kind"]), n_games=np.array(model["n_games"]),
             n_players=np.array(model["n_players"]), **{k: model[k] for k in model_keys(model)})
    os.replace(tmp, p)
    return {"file": _rel(p), "file_sha256": L.sha256_file(p), "arrays_sha256": model_sha(model),
            "n_games": model["n_games"], "n_players": model["n_players"]}


def load_model(rec):
    p = _abs(rec["file"])
    with np.load(p, allow_pickle=False) as z:
        model = {"kind": str(z["kind"]), "n_games": int(z["n_games"]), "n_players": int(z["n_players"])}
        for k in (LDA_KEYS if model["kind"] == "lda" else MLP_KEYS):
            model[k] = z[k]
    if model_sha(model) != rec["arrays_sha256"]:
        raise SystemExit("%s differs from tune.json" % rec["file"])
    return model


def project(X, model):
    return project_lda(X, model) if model["kind"] == "lda" else project_mlp(X, model)


# ================================================================ P1 terms
def zs_of(pools, Z, usable, pos):
    """The joint / b4 z term of each pool from per-game projections Z (= E43.p1_z with Zs['joint'] = Z)."""
    ctx = {"Zs": {P1_VARIANT: Z}, "usable": usable, "pos": pos}
    return [E43.p1_z(p, ctx) for p in pools]


def z_official(Z, usable, cand_rows, q_rows):
    """The joint / b4 z term on an official pool (groups = indices into Z), as P39E.p1_terms' joint part."""
    cg = [[g for g in grp if usable[g]] for grp in cand_rows]
    qg = [[g for g in grp if usable[g]] for grp in q_rows]
    T, has = P39E._term(Z, cg, qg, P1_PW)
    return st.zscore_rows(T, has), has


# ================================================================ seam
def cmd_seam(a):
    t0 = time.time()
    base = v26()
    rec = {"format": "c10-prereg45 seam v1", "base": base["record"], "git_head": L.git_head()}
    pools = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    Fs, vec = base_fused(base, "D", pools)
    ctx = E43.p1_ctx("D")
    if ctx["lda_sha256"] != base["lda_sha256"]:
        raise SystemExit("seam: the P1 LDA refit on L differs from prereg43 tune.json")
    per = seam_scores(base, "D", Fs, [E43.p1_z(p, ctx) for p in pools], pools)
    rec["D"] = {"per_pool": per, "mean": float(np.mean(per)), "want_mean": D_V26, "identical": True,
                "backend_only_identical": True, "vectors": vec, "lda_sha256": ctx["lda_sha256"], "sets": ctx["sets"]}
    # the blocked LDA (variant A's code) on L == fit_p1_lda on L, projections of the D games (no score computed)
    Lset = P39E.load_set("L")
    lab = np.array([g["player_id"] for g in Lset["games"]])
    mb = fit_lda_blocked(Lset["X"], lab, Lset["usable"], 64)
    Zb = project_lda(ctx["X"], mb)
    Zr = ctx["Zs"]["joint"]
    sign = np.sign(np.sum(Zb * Zr, 0))
    rec["blocked_lda_vs_fit_p1_lda_on_L"] = {
        "max_abs_diff_D_projection_sign_aligned": float(np.abs(Zb * sign - Zr).max()),
        "max_abs_D_projection": float(np.abs(Zr).max()), "sign_flips": int((sign < 0).sum()),
        "n_games": mb["n_games"], "n_players": mb["n_players"]}
    del Lset, mb, Zb, Zr, ctx, Fs, pools
    P39E._MOCK.clear()
    print(json.dumps({"D": rec["D"]["mean"], "blocked": rec["blocked_lda_vs_fit_p1_lda_on_L"]}), flush=True)
    if not a.skip_e:
        pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
        Fs, vec = base_fused(base, "E", pools)
        ctx = E43.p1_ctx("E")
        if ctx["lda_sha256"] != base["lda_sha256"]:
            raise SystemExit("seam: the P1 LDA refit on L differs from prereg43 tune.json")
        per = seam_scores(base, "E", Fs, [E43.p1_z(p, ctx) for p in pools], pools)
        rec["E"] = {"per_pool": per, "mean": float(np.mean(per)), "want_mean": E_V26, "identical": True,
                    "per_pool_identical_to_prereg43_eval": True, "backend_only_identical": True, "vectors": vec,
                    "sets": ctx["sets"]}
    rec["peak_working_set_mb"] = peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in rec if k != "base"}, indent=1))
    return 0


# ================================================================ select (D)
def no_e_results_yet():
    found = [p for p in (os.path.join(HERE, "eval.json"),) if os.path.exists(p)]
    if os.path.isdir(OUT_APPLY):
        found.append(OUT_APPLY)
    if found:
        raise SystemExit("prereg45 E / official results exist before select: %s" % [_rel(p) for p in found])


def cmd_select(a):
    t0 = time.time()
    no_e_results_yet()
    complete = {n: chunks_complete(n)[0] for n in ("M1", "M2")}
    names = check_sets_choice(a.sets, complete, datetime.datetime.now())
    base = v26()
    pools = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    Fs, vec = base_fused(base, "D", pools)
    ctx = E43.p1_ctx("D")                      # L-LDA64 (prereg39) = v2-6's P1 term; D seam first
    if ctx["lda_sha256"] != base["lda_sha256"]:
        raise SystemExit("the P1 LDA refit on L differs from prereg43 tune.json")
    base_per = seam_scores(base, "D", Fs, [E43.p1_z(p, ctx) for p in pools], pools)
    XD, uD, posD = ctx["X"], ctx["usable"], ctx["pos"]
    del ctx
    S = load_S(names)                          # S meta == L meta (load_S) == D meta (p1_ctx)
    P39E._MOCK.clear()                         # the PREREG_10 mock cache (~0.4 GB) is not needed any more
    print(json.dumps({"S": S["sets"], "seconds": round(time.time() - t0, 1)}), flush=True)
    out = {}
    for v in VARIANTS:
        t1 = time.time()
        if v in LDA_DIMS:
            model = fit_lda_blocked(S["X"], S["labels"], S["usable"], LDA_DIMS[v])
            info = {"dims": LDA_DIMS[v]}
        else:
            model, info = train_mlp(S["X"], S["labels"], S["usable"], consume=True)   # last: S["X"] is used up
            S["X"] = None
        mrec = save_model(v, model)
        Z = project(XD, model)
        del model
        zs = zs_of(pools, Z, uD, posD)
        del Z
        rows, top = E43.p1_grid(Fs, zs, pools)
        out[v] = {"model": mrec, "train": info, "grid": [{k: r[k] for k in ("mean", "wP", "k")} for r in rows],
                  "wP": top["wP"], "csls_k": top["k"], "mean": top["mean"], "per_pool": top["per_pool"],
                  "minus_v26_D": E43.diff_rec(top["per_pool"], base_per), "seconds": round(time.time() - t1, 1)}
        print(json.dumps({"variant": v, "mean": top["mean"], "wP": top["wP"], "k": top["k"],
                          "seconds": out[v]["seconds"]}), flush=True)
    c45 = first_max(list(VARIANTS), key=lambda x: out[x]["mean"])
    gate = out[c45]["mean"] - D_V26
    rec = {"format": "c10-prereg45 tune v1", "d_seeds": EV.D_SEEDS, "sets": list(names), "sets_arg": a.sets,
           "chunks_complete": complete, "S": S["sets"], "mlp_cfg": MLP, "variants_in_order": list(VARIANTS),
           "variants": out, "D": {v: out[v]["mean"] for v in VARIANTS},
           "c45": {"variant": c45, "model": out[c45]["model"], "variant_pw": [P1_VARIANT, P1_PW],
                   "wP": out[c45]["wP"], "csls_k": out[c45]["csls_k"], "mean": out[c45]["mean"],
                   "per_pool": out[c45]["per_pool"]},
           "cnn_vectors_D": vec, "base": base["record"],
           "base_D": {"per_pool": base_per, "mean": float(np.mean(base_per)), "identical_to_recorded": True},
           "gate": {"rule": "D(c45) - D(v2-6) >= %.2f, D(v2-6) = %r" % (GATE, D_V26), "value": gate,
                    "pass": bool(gate >= GATE)},
           "lists_record_sha256": L.sha256_file(RECORD), "peak_working_set_mb": peak_mb(),
           "seconds": round(time.time() - t0, 1), "git_head": L.git_head()}
    L.write_json(a.out, rec)
    print(json.dumps({"D": rec["D"], "c45": {k: rec["c45"][k] for k in ("variant", "wP", "csls_k", "mean")},
                      "gate": rec["gate"], "peak_working_set_mb": rec["peak_working_set_mb"]}, indent=1))
    return 0


# ================================================================ judge (E, once)
def e_judgements_so_far(repo=REPO):
    import glob
    return [d for d in COMP1_E_JUDGES if glob.glob(os.path.join(repo, "results", d, "explore", "eval*.json"))]


def judge_ready(tune, e_use, prior):
    if not tune["gate"]["pass"]:
        raise SystemExit("the D gate did not pass: the E pools are not used (PLAN.md 2)")
    if e_use != len(prior) + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before (%s)" % (e_use, len(prior), prior))


def cmd_judge(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    if not L.git_committed(a.tune):
        raise SystemExit("tune.json is not committed (or differs from HEAD); commit it before the judgement")
    tune = read_json(a.tune)
    prior = e_judgements_so_far()
    judge_ready(tune, a.e_use, prior)
    base = v26()
    if base["record"]["tune_sha256_lf"] != tune["base"]["tune_sha256_lf"] or \
            base["record"]["eval_sha256_lf"] != tune["base"]["eval_sha256_lf"]:
        raise SystemExit("v2-6's records changed since tune.json (PLAN.md 3: write a revision first)")
    model = load_model(tune["c45"]["model"])
    pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    Es = set(E43.splits()[0])
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in E43.store().pid[np.asarray(sorted(rows))].tolist()} <= Es:
            raise SystemExit("pool %d has players outside E" % p["seed"])
    Fs, vec = base_fused(base, "E", pools)
    ctx = E43.p1_ctx("E")
    if ctx["lda_sha256"] != base["lda_sha256"]:
        raise SystemExit("the P1 LDA refit on L differs from prereg43 tune.json")
    bper = seam_scores(base, "E", Fs, [E43.p1_z(p, ctx) for p in pools], pools)
    Z = project(ctx["X"], model)
    zs = zs_of(pools, Z, ctx["usable"], ctx["pos"])
    c = tune["c45"]
    per = E43.scores(Fs, zs, c["wP"], c["csls_k"], pools)
    byk = [EV.score_by_k(st.csls(F + c["wP"] * z, c["csls_k"]), p) for F, z, p in zip(Fs, zs, pools)]
    rec = {"format": "c10-prereg45 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": prior, "c45": c, "cnn_vectors_E": vec, "base": tune["base"],
           "seam_base_E_identical": True, "tune_sha256": st.sha256_lf(a.tune), "git_head": L.git_head(),
           "p1_sets": ctx["sets"], "means": {"c45": float(np.mean(per)), "base": float(np.mean(bper))},
           "per_pool": {"c45": per, "base": bper}, "by_k": {"c45": EV.mean_by_k(byk)}}
    j = E43.judge_rule(per, bper)
    rec["c45_minus_base"] = {x: j[x] for x in ("mean", "ci95", "pools_higher")}
    rec["rule"] = j["rule"].replace("c*", "c45")
    rec["adopt"] = j["adopt"]
    rec["peak_working_set_mb"] = peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "c45_minus_base", "by_k", "adopt")}, indent=1))
    return 0


# ================================================================ official pools
def cmd_apply(a):
    t0 = time.time()
    tune, ev = read_json(a.tune), read_json(a.eval)
    if not ev["adopt"] or ev["tune_sha256"] != st.sha256_lf(a.tune):
        raise SystemExit("eval.json does not adopt c45 (or belongs to another tune.json): keep v2-6")
    base = v26()
    if base["record"]["tune_sha256_lf"] != tune["base"]["tune_sha256_lf"]:
        raise SystemExit("v2-6's records changed since tune.json")
    model = load_model(tune["c45"]["model"])
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
    if P39E.models_sha(models) != base["lda_sha256"]:
        raise SystemExit("the L LDA refit differs from prereg43 tune.json")
    colour = np.array([c for _s, c in items])
    zb, _hb = P39E.p1_terms(Pset["X"], Pset["usable"], colour, rec["cand_rows"], rec["q_rows"], models,
                            P1_PW)[P1_VARIANT]
    del models
    cs = base["tune"]["c_star"]
    path, _choice = E43.cstar_file(base["tune"], a.pool)
    e = E43.pool_emb(path, items, cs["ckpt_sha256"])
    F = BK.fuse_cfg(rec, BK.e_term(rec, e, base["cfg"]), base["cfg"])
    # seam: v2-6 rebuilt on this pool == prereg43's uploaded top5
    data_b = P39E.top5_bytes(qids, st.top5(st.csls(F + base["wP"] * zb, base["k"]), cand_ids))
    with open(os.path.join(V26_APPLY, a.pool, "top5.csv"), "rb") as f:
        if f.read() != data_b:
            raise SystemExit("seam: v2-6 rebuilt on %s differs from results/prereg43/apply/%s/top5.csv"
                             % (a.pool, a.pool))
    Z = project(Pset["X"], model)
    z, has = z_official(Z, Pset["usable"], rec["cand_rows"], rec["q_rows"])
    c = tune["c45"]
    data = P39E.top5_bytes(qids, st.top5(st.csls(F + c["wP"] * z, c["csls_k"]), cand_ids))
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as f:
        f.write(data)
    G = BK.pool_games(e, rec)
    rep = {"format": "c10-prereg45 apply v1", "pool": a.pool, "c45": c, "vectors": _rel(path),
           "vectors_sha256": L.sha256_file(path), "cfg": base["cfg"], "v26_top5_equals_prereg43_apply": True,
           "questions": len(qids), "candidates": len(cand_ids), "games": len(items),
           "games_without_vector": int((e.count == 0).sum()),
           "candidates_without_vector": int(G["nc"] - len(np.unique(G["gc"]))),
           "questions_without_vector": int(G["nq"] - len(np.unique(G["gq"]))),
           "games_without_P1": bad, "questions_without_P1": int((~np.asarray(has)).sum()),
           "tune_sha256": st.sha256_lf(a.tune), "eval_sha256": st.sha256_lf(a.eval),
           "top5_sha256": L.sha256_file(out), "differs_from_v26_rows": int(sum(
               x != y for x, y in zip(data.split(b"\n"), data_b.split(b"\n")))),
           "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    return E43.cmd_assemble(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("seam")
    m.add_argument("--skip-e", action="store_true")
    m.add_argument("--out", default=os.path.join(HERE, "seam.json"))
    s = sub.add_parser("select")
    s.add_argument("--sets", required=True, choices=sorted(SET_CHOICES), help="PLAN.md 1 (completion only)")
    s.add_argument("--out", default=os.path.join(HERE, "tune.json"))
    j = sub.add_parser("judge")
    j.add_argument("--e-use", type=int, required=True)
    j.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    j.add_argument("--out", default=os.path.join(HERE, "eval.json"))
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    p.add_argument("--eval", default=os.path.join(HERE, "eval.json"))
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", default=V1_SUBMISSION)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd != "assemble":
        BK.polite()                 # BELOW_NORMAL, 4 logical processors, 1 BLAS thread (prereg38's manners)
        BK.wait_for_ram(need_mb=RAM_NEED_MB)
    return {"seam": cmd_seam, "select": cmd_select, "judge": cmd_judge, "apply": cmd_apply,
            "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
