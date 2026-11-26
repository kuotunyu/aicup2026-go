"""PREREG_09 competition 2: the P3 manifest, the prepared probe / inner arrays, the selfcheck and the training of the
ens3fam members (docs/PREREG_09_MAPPING_ENSEMBLE.md rev 1, sections 1.2-1.4, 2.3; steps 3-5 of section 9).

    PY = .venv/Scripts/python.exe   (CPU; Git Bash; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL thread variable)
    PY tools/p09_map.py manifest [--precommit] [--rerun --reason T]      -> results/prereg09/feat/p3_manifest.json
    PY tools/p09_map.py prep     [--rerun --reason T]                    -> data/prereg09/map/ (git-ignored)
    PY tools/p09_map.py selfcheck [--precommit] [--rerun --reason T]     -> results/prereg09/rank/selfcheck.json
                                                                            (+ data/prereg09/map/selfcheck/)
    PY tools/p09_map.py train    [--fallback] [--precommit]              -> data/prereg09/map/train/ (fallback/) +
                                                                            results/prereg09/rank/train_report.json
                                                                            (train_fallback_report.json)
    PY tools/p09_map.py status                                           (read only)
    preview: prep --out-dir data/prereg09/tmp/<x> [--probe P1+P2] [--limit-chunks K];
             selfcheck / train --out-dir <same> [--threads T] (a preview never feeds a real step)

manifest  Only when P3 has all 600 chunks or after 2026-10-10 23:59:59 (+08:00) (PREREG_09 1.2). P3 chunks
          data/prereg09/p3/probe: per chunk SHA-256, games, failed, no target move, refs equal to the sets.json
          segment, metadata (format, sides, precision, device, batch, channels_last, neutral-metadata SHA, KataGo commit,
          p1_lib / p1_extract SHA); p3 = {complete, chunks, failed, used, reason} by the fixed rule (600 / 600 valid
          and failed <= 600). disjoint: P3 players vs inner and validation players, P3 games vs P1 / P2 probe games.
          duplicates: P3 games whose move sequence (original orientation) equals a bank / mono or P1 / P2 probe
          game's (never used). Records the LF SHA-256 of the exploration scripts. Reads chunk metadata and counts,
          SHA-256 and the training SGFs only.
prep      After the committed manifest: P1 (+ P2) (+ P3 when used) probe games (usable, not a duplicate; chunk SHA-256
          against 08's inner_manifest.json / p3_manifest.json while reading) and every usable bank / mono game ->
          float32 raw arrays (X with NaN opponent halves, EX raw; tools/p09_forward.py) + meta + prep_record.json.
selfcheck PREREG_09 2.3 after check: 1 fusion seam (Q02H, Q08, the exploration's six lp_inner averaged x 4: within
          1e-12); 2 retrain the six members on P1+P2 with the registered recipe; numpy vs torch forward <= 1e-4;
          ensemble inner within 0.002 of 0.7113584160645174 (otherwise stop). Record only: CV log-loss, e*, the
          maximum lp difference to the exploration.
train     PREREG_09 1.3 on the probe the manifest rule chose, 8 torch threads; resumable per member (a member whose
          file exists and matches is kept). --fallback: Q08P's LR (p1_fuse.fit_mapping_grouped, PREREG_08 1.5 grid),
          only after Q09's tune_report.json says the gates did not pass.
Exit codes: 0 done; 1 the step ran and says stop; 2 refused or error.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import p09_forward as FW  # noqa: E402

L = FW.L
BATCH = 512
PREDICT_BATCH = 8192
FOLDS = 5
# functions of this file that may import u1_lib / call a mapping fit (tests/test_p09_hygiene.py checks)
INNER_ONLY_FUNCS = ("cmd_selfcheck", "_fit_fallback", "cmd_train")


def R9():
    """tools/p09_rank_fuse.py (pins the p08 tools before importing them)."""
    import p09_rank_fuse as r
    return r


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ================================================================ torch members (port of explore/mapping/train.py)
_CLASSES = {}


def torch_classes():
    """The three member classes of train.py (MLPCLM, MLP, LowRank), unchanged."""
    if _CLASSES:
        return _CLASSES
    import torch

    def clm_logp(f, t0, dt):
        th = torch.cat([t0.view(1), t0 + torch.cumsum(torch.nn.functional.softplus(dt), 0)])
        c = torch.sigmoid(th.view(1, -1) - f)
        c = torch.cat([torch.zeros_like(f), c, torch.ones_like(f)], 1)
        return torch.log((c[:, 1:] - c[:, :-1]).clamp_min(1e-12))

    class LowRank(torch.nn.Module):
        def __init__(self, d, r):
            super().__init__()
            self.a = torch.nn.Linear(d, r, bias=False)
            self.b = torch.nn.Linear(r, L)
            torch.nn.init.normal_(self.a.weight, std=0.01)

        def forward(self, x):
            return torch.log_softmax(self.b(self.a(x)), 1)

        def penalty(self):
            return (self.a.weight ** 2).sum() + (self.b.weight ** 2).sum()

    class MLPCLM(torch.nn.Module):
        def __init__(self, d, hidden, drop_in, drop):
            super().__init__()
            layers = [torch.nn.Dropout(drop_in)]
            prev = d
            for h in hidden:
                layers += [torch.nn.Linear(prev, h), torch.nn.GELU(), torch.nn.Dropout(drop)]
                prev = h
            layers.append(torch.nn.Linear(prev, 1))
            self.f = torch.nn.Sequential(*layers)
            self.t0 = torch.nn.Parameter(torch.tensor(-4.0))
            self.dt = torch.nn.Parameter(torch.full((L - 2,), math.log(math.e - 1)))

        def forward(self, x):
            return clm_logp(self.f(x), self.t0, self.dt)

        def penalty(self):
            return torch.zeros(())

    class MLP(torch.nn.Module):
        def __init__(self, d, hidden, drop_in, drop):
            super().__init__()
            layers = [torch.nn.Dropout(drop_in)]
            prev = d
            for h in hidden:
                layers += [torch.nn.Linear(prev, h), torch.nn.GELU(), torch.nn.Dropout(drop)]
                prev = h
            layers.append(torch.nn.Linear(prev, L))
            self.f = torch.nn.Sequential(*layers)

        def forward(self, x):
            return torch.log_softmax(self.f(x), 1)

        def penalty(self):
            return torch.zeros(())

    _CLASSES.update(torch=torch, LowRank=LowRank, MLPCLM=MLPCLM, MLP=MLP)
    return _CLASSES


def make_model(kind, d, seed):
    """train.py make(): torch.manual_seed(seed), then the module."""
    T = torch_classes()
    T["torch"].manual_seed(seed)
    cfg = FW.CONFIGS[kind]
    if kind == "lowrank":
        return T["LowRank"](d, cfg["rank"])
    if kind == "mlpclm":
        return T["MLPCLM"](d, cfg["hidden"], cfg["drop_in"], cfg["drop"])
    if kind == "mlp":
        return T["MLP"](d, cfg["hidden"], cfg["drop_in"], cfg["drop"])
    raise ValueError(kind)


class Rows:
    """Float32 training input of a probe subset: float32(standardised float64 X) (+ EX) from the raw arrays,
    = what prep.py stored and train.py read."""

    def __init__(self, X, EX, mom, with_ex):
        self.X, self.EX, self.mom, self.with_ex = X, EX, mom, with_ex
        self.d = FW.X_DIM + (FW.NEX if with_ex else 0)

    def __call__(self, r):
        xb = FW.standardise_x(np.asarray(self.X[r], dtype=np.float64), self.mom["mu_x"], self.mom["sd_x"])
        xb = xb.astype(np.float32)
        if self.with_ex:
            eb = FW.standardise_ex(self.EX[r], self.mom["mu_ex"], self.mom["sd_ex"]).astype(np.float32)
            xb = np.concatenate([xb, eb], 1)
        return xb


def batches(rows, get, bs, rng, shuffle=True):
    rows = rng.permutation(rows) if shuffle else rows
    for s in range(0, len(rows), bs):
        r = np.sort(rows[s:s + bs])
        yield r, torch_classes()["torch"].from_numpy(get(r))


def predict(model, get, n, rows=None, bs=PREDICT_BATCH):
    torch = torch_classes()["torch"]
    model.eval()
    rows = np.arange(n) if rows is None else rows
    out = np.zeros((len(rows), L))
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            r = rows[s:s + bs]
            out[s:s + len(r)] = model(torch.from_numpy(get(r))).double().numpy()
    return out


def train_one(kind, get, d, y, rows, seed, epochs, stop_after=None, eval_rows=None, n=None):
    """train.py train() for the three registered kinds: OneCycle over `epochs` (pct_start 0.1), batch 512, rows
    shuffled per epoch with default_rng(seed) and sorted within a batch; stops after `stop_after` epochs. With
    eval_rows: the per-epoch float32 probabilities of those rows."""
    torch = torch_classes()["torch"]
    cfg = FW.CONFIGS[kind]
    model = make_model(kind, d, seed)
    if kind in ("mlp", "mlpclm"):
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    steps = epochs * math.ceil(len(rows) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    yt = torch.from_numpy(y)
    snaps = []
    for _ep in range(stop_after or epochs):
        model.train()
        for r, xb in batches(rows, get, BATCH, rng):
            lp = model(xb)
            loss = torch.nn.functional.nll_loss(lp, yt[r])
            if kind not in ("mlp", "mlpclm"):
                loss = loss + cfg["lam"] * model.penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if eval_rows is not None:
            snaps.append(np.exp(predict(model, get, n, eval_rows)).astype(np.float32))
    return model, snaps


def rule_b_score(lp, y):
    p = np.exp(lp)
    left = np.zeros_like(p)
    right = np.zeros_like(p)
    left[:, 1:] = p[:, :-1]
    right[:, :-1] = p[:, 1:]
    pred = np.argmax(p + math.exp(-1) * (left + right), 1)
    dd = np.abs(pred - y)
    return np.where(dd == 0, 1.0, np.where(dd == 1, math.exp(-1), 0.0))


def fit_member(spec, get, d, y, fold, log=print):
    """PREREG_09 1.3 items 2-3 for one member (= train.py run() with one seed): player-grouped 5-fold CV, per-epoch
    OOF for the MLP kinds (e* by the mean OOF log-loss), refit on every row stopped at e* (lowrank: fixed 30).
    -> (model, record)."""
    kind, seed = spec["kind"], spec["seed"]
    cfg = FW.CONFIGS[kind]
    epochs = cfg["epochs"]
    n = len(y)
    is_mlp = kind in ("mlp", "mlpclm")
    E = epochs if is_mlp else 1
    oof_p = np.zeros((E, n, L), dtype=np.float32)
    t0 = time.time()
    for f in range(FOLDS):
        tr, te = np.nonzero((fold != f) & (fold >= 0))[0], np.nonzero(fold == f)[0]
        if is_mlp:
            _m, snaps = train_one(kind, get, d, y, tr, seed, epochs, eval_rows=te, n=n)
            for e in range(E):
                oof_p[e, te] += snaps[e] / 1
        else:
            model, _ = train_one(kind, get, d, y, tr, seed, epochs)
            oof_p[0, te] += (np.exp(predict(model, get, n, te)) / 1).astype(np.float32)
        lt = np.log(np.clip(oof_p[-1, te].astype(np.float64), 1e-300, None))
        log(f"{spec['name']} fold {f}: CV ll (last epoch) {-lt[np.arange(len(te)), y[te]].mean():.4f} "
            f"{time.time() - t0:.0f}s")
    sub = np.nonzero(fold >= 0)[0]
    best, ll_ep = R9().epoch_rule(oof_p[:, sub] if len(sub) != n else oof_p, y[sub])
    best_ep = best if is_mlp else epochs
    oof = np.log(np.clip(oof_p[best_ep - 1 if is_mlp else 0].astype(np.float64), 1e-300, None))
    del oof_p
    cv = {"logloss": float(-oof[sub, y[sub]].mean()), "top1": float((oof[sub].argmax(1) == y[sub]).mean()),
          "rule_b_score_per_game": float(rule_b_score(oof[sub], y[sub]).mean()), "games": int(len(sub)),
          "logloss_by_epoch": ll_ep if is_mlp else None, "epochs_chosen": best_ep, "epochs_schedule": epochs,
          "seconds_cv": round(time.time() - t0, 1)}
    log(f"{spec['name']}: CV {cv['logloss']:.4f} top1 {cv['top1']:.4f} epochs {best_ep}/{epochs}")
    t1 = time.time()
    model, _ = train_one(kind, get, d, y, sub, seed, epochs, stop_after=best_ep if is_mlp else None)
    cv["seconds_refit"] = round(time.time() - t1, 1)
    return model, cv


def model_params(model, kind):
    sd = model.state_dict()
    keys = FW.STATE_KEYS[kind]
    if sorted(sd) != sorted(keys):
        raise RuntimeError(f"{kind}: state_dict keys {sorted(sd)}")
    return {keys[k]: sd[k].detach().cpu().numpy().astype(np.float32) for k in keys}


def model_from_params(kind, params):
    torch = torch_classes()["torch"]
    d = FW.X_DIM + (FW.NEX if FW.uses_ex(kind) else 0)
    model = make_model(kind, d, 0)
    inv = {v: k for k, v in FW.STATE_KEYS[kind].items()}
    model.load_state_dict({inv[p]: torch.from_numpy(np.asarray(v, dtype=np.float32).copy())
                           for p, v in params.items()})
    model.eval()
    return model


def torch_member_logp(kind, params, X, EX=None, double=False):
    """The torch module's eval-mode lp: float32 (as trained) or, with double=True, the same stored float32 weights
    cast to float64 with float64 inputs."""
    torch = torch_classes()["torch"]
    model = model_from_params(kind, params)
    dt = np.float64 if double else np.float32
    if double:
        model = model.double()
    xb = X if not FW.uses_ex(kind) else np.concatenate([X, EX], 1)
    with torch.no_grad():
        return model(torch.from_numpy(np.ascontiguousarray(xb, dtype=dt))).double().numpy()


def torch_numpy_max_diff(map_path, Xraw, EXraw, block=8192):
    """PREREG_09 2.3 / 2.6: the same weights through torch and through the numpy float64 forward.
    Judged (max_abs_lp_diff): torch in float64 (the stored float32 weights cast to float64, float64 standardised
    input) vs numpy, ensemble lp: a check of the port (formula, layouts, GELU, softplus, cumulative link).
    Record only (float32): torch in float32 as trained (float32 input) vs numpy, lp and probability differences; the
    cumulative link's upper tail (differences of sigmoids near 1) loses relative precision in float32, so its lp
    difference is not a port check."""
    mp = FW.load_map(map_path)
    worst, per = 0.0, {m["name"]: 0.0 for m in mp["members"]}
    w32, p32, per32 = 0.0, 0.0, {m["name"]: 0.0 for m in mp["members"]}
    for s in range(0, len(Xraw), block):
        X64 = FW.standardise_x(np.asarray(Xraw[s:s + block], dtype=np.float64), mp["mu_x"], mp["sd_x"])
        E64 = FW.standardise_ex(EXraw[s:s + block], mp["mu_ex"], mp["sd_ex"])
        if not len(X64):
            continue
        lp_np, lps_np = FW.ensemble_logp([(m["kind"], m["params"]) for m in mp["members"]], X64, E64)
        lps_t = [torch_member_logp(m["kind"], m["params"], X64, E64, double=True) for m in mp["members"]]
        worst = max(worst, float(np.max(np.abs(FW.lse_mean(lps_t) - lp_np))))
        for m, a, b in zip(mp["members"], lps_t, lps_np):
            per[m["name"]] = max(per[m["name"]], float(np.max(np.abs(a - b))))
        lps_32 = [torch_member_logp(m["kind"], m["params"], X64.astype(np.float32), E64.astype(np.float32))
                  for m in mp["members"]]
        lp_32 = FW.lse_mean(lps_32)
        w32 = max(w32, float(np.max(np.abs(lp_32 - lp_np))))
        p32 = max(p32, float(np.max(np.abs(np.exp(lp_32) - np.exp(lp_np)))))
        for m, a, b in zip(mp["members"], lps_32, lps_np):
            per32[m["name"]] = max(per32[m["name"]], float(np.max(np.abs(a - b))))
    return {"max_abs_lp_diff": worst, "per_member": per, "games": int(len(Xraw)),
            "definition": "torch float64 forward (eval; stored float32 weights cast to float64) vs numpy float64 "
                          "forward, ensemble lp (judged)",
            "float32_record_only": {"max_abs_lp_diff": w32, "max_abs_prob_diff": p32, "per_member_lp": per32,
                                    "definition": "torch float32 forward as trained (float32 input) vs numpy"}}


def set_threads(n):
    torch = torch_classes()["torch"]
    torch.set_num_threads(int(n))
    return torch.get_num_threads()


def save_member(path, spec, params, cv, label, rows, extra):
    arrays = {"format": np.array(R9().FORMATS["member"]), "name": np.array(spec["name"]), "kind": np.array(spec["kind"]),
              "seed": np.array(spec["seed"]), "e_star": np.array(cv["epochs_chosen"]), "cv": np.array(json.dumps(cv)),
              "probe_used": np.array(label), "rows": np.array(rows), "extra": np.array(json.dumps(extra, sort_keys=True))}
    arrays.update({f"p_{k}": v for k, v in params.items()})
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def member_binding(prep, lo, hi, threads):
    """What a saved member file must match to be reused (M1): the prepared arrays and rows, the training code (LF
    SHA-256 of tools/p09_map.py and tools/p09_forward.py), the torch version and the torch thread count."""
    r = R9()
    import torch
    return {"prep_record_sha256": prep.record_sha, "rows_first_last": [int(lo), int(hi)],
            "code_sha256": {f: r.sha_lf(r.path_of(f)) for f in r.TRAINING_CODE}, "torch": torch.__version__,
            "threads": int(threads)}


def member_reusable(path, spec, label, rows, extra):
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            got = (str(z["name"]), str(z["kind"]), int(z["seed"]), str(z["probe_used"]), int(z["rows"]),
                   json.loads(str(z["extra"])))
    except Exception:  # noqa: BLE001
        return False
    return got == (spec["name"], spec["kind"], spec["seed"], label, rows, json.loads(json.dumps(extra,
                                                                                               sort_keys=True)))


def train_members(prep, sets, folder, threads, log=print, members=None):
    """Train the ens3fam members on the probe rows of `sets` -> (moments, member records). Each member file is
    written as soon as it is done (a rerun keeps it when name, kind, seed, probe and binding match)."""
    r = R9()
    idx = prep.subset(sets)
    label = r.probe_used_label(sets)
    os.makedirs(folder, exist_ok=True)
    mom = prep.moments(idx)
    r.save_moments(os.path.join(folder, "moments.npz"), mom, label)
    y = prep.y[idx]
    players = prep.players[idx]
    fold = FW.player_folds(players)
    if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))) if len(idx) else False:
        raise r.P09Error("probe rows of the subset are not contiguous")
    lo, hi = (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)
    Xv, EXv = prep.probe_X[lo:hi], prep.probe_ex[lo:hi]
    used_threads = set_threads(threads)
    extra = member_binding(prep, lo, hi, used_threads)
    recs = {}
    for spec in FW.MEMBERS:
        if members and spec["name"] not in members:
            continue
        path = os.path.join(folder, f"member_{spec['name']}.npz")
        if member_reusable(path, spec, label, len(idx), extra):
            m = r.load_member(path)
            recs[spec["name"]] = {"cv": m["cv"], "e_star": m["e_star"], "reused": True}
            log(f"{spec['name']}: kept {os.path.basename(path)}")
            continue
        with_ex = FW.uses_ex(spec["kind"])
        get = Rows(Xv, EXv, mom, with_ex)
        t = time.time()
        model, cv = fit_member(spec, get, get.d, y, fold, log=log)
        params = model_params(model, spec["kind"])
        save_member(path, spec, params, cv, label, len(idx), extra)
        recs[spec["name"]] = {"cv": cv, "e_star": cv["epochs_chosen"], "seconds": round(time.time() - t, 1),
                              "reused": False}
    return mom, recs, {"probe_used": label, "rows": int(len(idx)), "players": int(len(set(players.tolist()))),
                       "folds": {str(f): int((fold == f).sum()) for f in range(FOLDS)}, "threads": used_threads}


def folder_files(folder):
    return {f: R9().sha_raw(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
            if f.endswith(".npz") and ".tmp" not in f}


# ================================================================ manifest (PREREG_09 1.2, 2.2 item 1)
def p3_presence():
    r = R9()
    present = sum(1 for ci in range(r.P3_CHUNKS) if os.path.exists(os.path.join(r.P3_DIR, f"chunk_{ci:05d}.npz")))
    return present, r.P3_CHUNKS


def scan_p3(games):
    """P3 chunk records against the P3 sets.json games (as p08_p1.scan_set)."""
    r = R9()
    R = r.R
    import p08_p1
    refs_all = np.array([g["ref"] for g in games], dtype=np.int64)
    plan = p08_p1.chunk_plan(len(games), R.RUN["chunk"])
    rows = []
    tot = {"games": 0, "failed": 0, "no_target_moves": 0, "no_move": 0, "failed_or_no_move": 0}
    refs_ok = meta_ok = True
    for ci, st, en in plan:
        p = os.path.join(r.P3_DIR, f"chunk_{ci:05d}.npz")
        if not os.path.exists(p):
            continue
        rec, ref, _failed, _cnt, _nm = R.chunk_meta(p)
        rec["ref_matches_sets"] = bool(np.array_equal(ref, refs_all[st:en]))
        rec["meta_problems"] = R.chunk_meta_problems(rec, "probe")
        refs_ok = refs_ok and rec["ref_matches_sets"]
        meta_ok = meta_ok and not rec["meta_problems"]
        for k in tot:
            tot[k] += rec[k]
        rows.append(rec)
    names = {f"chunk_{ci:05d}.npz" for ci, _s, _e in plan}
    extra = sorted(r.rel(f) for f in R.chunk_files(r.P3_DIR) if os.path.basename(f) not in names)
    return {"chunks": rows, "chunks_expected": len(plan), "chunks_present": len(rows), "games_planned": len(games),
            "registered_chunks": r.P3_CHUNKS, "registered_games": r.P3_GAMES, "totals": tot,
            "refs_match_sets": bool(refs_ok), "meta_matches": bool(meta_ok), "extra_files": extra,
            "complete": bool(len(rows) == len(plan) == r.P3_CHUNKS and len(games) == r.P3_GAMES and refs_ok
                             and meta_ok and not extra)}


def cmd_manifest(a):
    t0 = time.time()
    r = R9()
    R = r.R
    out = r.MANIFEST
    if os.path.exists(out):
        if not (a.rerun and a.reason):
            raise r.P09Error(f"{r.rel(out)} exists; rerun only with --rerun --reason TEXT")
        if r.git(["ls-files", "--error-unmatch", "--", r.rel(out)])[0] == 0:
            raise r.P09Error("p3_manifest.json is committed; it is never overwritten")
    r.blas_env_check(False)
    ts = r.tool_state()
    r.require_tools(ts, a.precommit)
    pins = r.check_pins()
    now = datetime.datetime.now(datetime.timezone.utc)
    present, expected = p3_presence()
    if present < expected and now <= r.P3_CUTOFF:            # early refusal (no chunk is read)
        r.p3_decision(present, expected, False, 0, now)
    L1, _F, E = R.import_p1_extract()
    sets1, sets2, sets3 = E.load_sets(R.P1_SETS), E.load_sets(R.P2_SETS), E.load_sets(r.P3_SETS)
    t = time.time()
    games3 = E.set_games(sets3, "probe")
    srec = scan_p3(games3)
    print(f"p3_probe: {srec['chunks_present']}/{srec['chunks_expected']} chunks, {srec['totals']} "
          f"({time.time() - t:.0f}s)", flush=True)
    p3 = r.p3_decision(srec["chunks_present"], expected,
                       bool(srec["refs_match_sets"] and srec["meta_matches"] and not srec["extra_files"]
                            and srec["games_planned"] == r.P3_GAMES), srec["totals"]["failed"], now)
    p3["games"] = srec["totals"]["games"]
    p3["no_target_moves_record_only"] = srec["totals"]["no_target_moves"]
    # disjoint (PREREG_09 1.2, 2.2 item 4)
    import c10data
    split = c10data.read_split(R.SPLIT_CSV)
    inner_p = c10data.inner_players(split)
    val_p = {p for p, info in split.items() if info["split"] != "train"}
    pl3 = {g["player_id"] for g in sets3["probe"]["games"]}
    g12 = {(g["rank"], g["game_id"]) for g in sets1["probe"]["games"]} | \
          {(g["rank"], g["game_id"]) for g in sets2["probe"]["games"]}
    g3 = {(g["rank"], g["game_id"]) for g in sets3["probe"]["games"]}
    dj = {"probe_players_p3": len(pl3), "inner_overlap": len(pl3 & inner_p), "validation_overlap": len(pl3 & val_p),
          "p3_p1p2_game_overlap": len(g3 & g12)}
    dj["passed"] = dj["inner_overlap"] == 0 and dj["validation_overlap"] == 0 and dj["p3_p1p2_game_overlap"] == 0
    # duplicates by move sequence (original orientation): P3 vs bank / mono and P1 / P2 probe
    import p08_p1
    t = time.time()
    h3 = [p08_p1.move_hash(g["sgf_content"]) for g in games3]
    other_inner, other_probe = set(), set()
    for sj, name, bucket in ((sets1, "bank", other_inner), (sets1, "mono", other_inner), (sets1, "probe", other_probe),
                             (sets2, "probe", other_probe)):
        bucket |= {p08_p1.move_hash(g["sgf_content"]) for g in E.set_games(sj, name)}
    refs3 = [g["ref"] for g in games3]
    d_inner = p08_p1.duplicate_refs(refs3, h3, other_inner)
    d_probe = p08_p1.duplicate_refs(refs3, h3, other_probe)
    dup = {"p3_probe": sorted(set(d_inner) | set(d_probe)), "vs_bank_mono": d_inner, "vs_p1_p2_probe": d_probe,
           "count": {"p3_probe": len(set(d_inner) | set(d_probe)), "vs_bank_mono": len(d_inner),
                     "vs_p1_p2_probe": len(d_probe)},
           "p3_games_without_moves": sum(1 for x in h3 if x is None),
           "within_p3_repeats_record_only": len([x for x in h3 if x is not None]) - len({x for x in h3 if x}),
           "method": "sha256 of p04_opp.encode_one(sgf) (move sequence, original orientation); P3 refs whose hash "
                     "equals a bank / mono or P1 / P2 probe game's are never used (PREREG_09 1.2)",
           "seconds": round(time.time() - t, 1)}
    del games3, h3
    man = {"format": r.FORMATS["manifest"], "what": "PREREG_09 1.2 / 2.2: P3 chunk manifest (no score)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), "precommit": bool(a.precommit),
           "p3": p3, "p3_chunks": srec, "disjoint": dj, "duplicates": dup,
           "sets_sha256": {"data/prereg09/p3/sets.json": r.sha_raw(r.P3_SETS),
                           "data/prereg08/p1/sets.json": r.sha_raw(R.P1_SETS),
                           "data/prereg08/p2/sets.json": r.sha_raw(R.P2_SETS)},
           "exploration_scripts_sha256_lf": {k: pins[k] for k in r.EXPLORE_PINS},
           "manifest_code_sha256": {"tools/p09_map.py": r.sha_lf(os.path.abspath(__file__)),
                                    "results/prereg08/p1/p1_lib.py": pins["results/prereg08/p1/p1_lib.py"],
                                    "results/prereg08/p1/p1_extract.py": pins["results/prereg08/p1/p1_extract.py"]},
           "neutral_metadata_sha256": L1.neutral_table_sha(), "torch_values": sorted({c["torch"] for c in
                                                                                       srec["chunks"]}),
           "pinned_sha256": pins, "tool": ts, "environment": r.environment_cpu()}
    r.write_json(out, man)
    print(f"manifest: P3 used {p3['used']} ({p3['reason']}); duplicates {dup['count']}; disjoint {dj['passed']} -> "
          f"{r.rel(out)}")
    return 0


# ================================================================ prep (PREREG_09 1.3 item 1)
def read_chunk_rows(path, want_sha, dups, games=None):
    """One P1-format chunk -> usable, non-duplicate rows: (ref, X raw float32, EX raw float32, y, players, sha)."""
    r = R9()
    sha = r.sha_raw(path)
    if want_sha is not None and sha != want_sha:
        raise r.P09Error(f"{r.rel(path)}: SHA-256 differs from the manifest")
    with np.load(path, allow_pickle=False) as z:
        if str(z["format"]) != r.R.CHUNK_FORMAT or str(z["meta_sha256"]) != r.R.NEUTRAL_META_SHA:
            raise r.P09Error(f"{r.rel(path)}: format / metadata")
        ref = np.asarray(z["ref"]).astype(np.int64)
        failed = np.asarray(z["failed"], dtype=bool)
        counts = np.asarray(z["counts"])
        us = FW.usable_games(failed, counts)
        keep = us & ~np.isin(ref, np.asarray(sorted(dups), dtype=np.int64))
        G = len(ref)
        X = FW.x_raw32(np.asarray(z["means"]), counts)
        EX = FW.ex_raw32({k: np.asarray(z[k]) for k in ("pos_game", "pos_side", "pos_turn", "pos_logp",
                                                         "pos_entropy")}, G)
    out = {"ref": ref[keep], "X": X[keep], "EX": EX[keep], "sha": sha, "games": G, "unusable": int((~us).sum()),
           "dups": int((us & ~keep).sum())}
    if games is not None:
        lv = {x: i for i, x in enumerate(FW_LEVELS())}
        out["y"] = np.array([lv[games[j]["rank"]] for j in out["ref"]], dtype=np.int64)
        out["players"] = np.array([games[j]["rank"] + ":" + games[j]["player_id"] for j in out["ref"]])
    return out


def FW_LEVELS():
    return list(R9().LEVELS)


def usable_count(path, dups):
    with np.load(path, allow_pickle=False) as z:
        ref = np.asarray(z["ref"]).astype(np.int64)
        us = FW.usable_games(np.asarray(z["failed"], dtype=bool), np.asarray(z["counts"]))
    return int((us & ~np.isin(ref, np.asarray(sorted(dups), dtype=np.int64))).sum())


def build_prep(map_dir, probe_parts, inner_parts, record, log=print):
    """probe_parts: [(set name, [(chunk path, SHA or None)], sets.json probe games, duplicate refs)] in order;
    inner_parts: [(set name, [(chunk path, SHA or None)])]. Writes the raw arrays, meta and prep_record.json."""
    r = R9()
    os.makedirs(map_dir, exist_ok=True)
    t0 = time.time()
    sizes = {}
    for name, files, _g, dups in probe_parts:
        sizes[name] = sum(usable_count(p, dups) for p, _s in files)
    for name, files in inner_parts:
        sizes[name] = sum(usable_count(p, ()) for p, _s in files)
    n_probe = sum(sizes[n] for n, *_ in probe_parts)
    n_inner = sum(sizes[n] for n, _f in inner_parts)
    rec_sets = {}

    def stream(prefix, parts, n, with_labels):
        X = np.lib.format.open_memmap(os.path.join(map_dir, f"{prefix}_X.npy.tmp"), mode="w+", dtype=np.float32,
                                      shape=(n, FW.X_DIM))
        EX = np.lib.format.open_memmap(os.path.join(map_dir, f"{prefix}_ex.npy.tmp"), mode="w+", dtype=np.float32,
                                       shape=(n, FW.NEX))
        meta = {"ref": [], "src": [], "y": [], "players": []}
        i = 0
        for part in parts:
            name, files = part[0], part[1]
            games = part[2] if with_labels else None
            dups = set(int(x) for x in (part[3] if with_labels else ()))
            srec = {"chunks": 0, "games_in_chunks": 0, "unusable": 0, "duplicates_excluded": 0, "used": 0,
                    "chunk_sha256": []}
            for k, (p, want) in enumerate(files):
                c = read_chunk_rows(p, want, dups, games)
                g = len(c["ref"])
                X[i:i + g] = c["X"]
                EX[i:i + g] = c["EX"]
                meta["ref"].append(c["ref"])
                meta["src"] += [name] * g
                if with_labels:
                    meta["y"].append(c["y"])
                    meta["players"].append(c["players"])
                i += g
                srec["chunks"] += 1
                srec["games_in_chunks"] += c["games"]
                srec["unusable"] += c["unusable"]
                srec["duplicates_excluded"] += c["dups"]
                srec["used"] += g
                srec["chunk_sha256"].append(c["sha"])
                if k % 100 == 0:
                    log(f"  {prefix} {name} {k}/{len(files)} chunks, {i} rows, {time.time() - t0:.0f}s")
            rec_sets[name] = srec
        if i != n:
            raise r.P09Error(f"{prefix}: wrote {i} rows, expected {n}")
        X.flush()
        EX.flush()
        del X, EX
        os.replace(os.path.join(map_dir, f"{prefix}_X.npy.tmp"), os.path.join(map_dir, f"{prefix}_X.npy"))
        os.replace(os.path.join(map_dir, f"{prefix}_ex.npy.tmp"), os.path.join(map_dir, f"{prefix}_ex.npy"))
        out = {"ref": np.concatenate(meta["ref"]) if meta["ref"] else np.zeros(0, np.int64),
               "src": np.array(meta["src"], dtype=str)}
        if with_labels:
            out["y"] = np.concatenate(meta["y"]) if meta["y"] else np.zeros(0, np.int64)
            out["players"] = np.concatenate(meta["players"]) if meta["players"] else np.zeros(0, dtype=str)
        np.savez(os.path.join(map_dir, f"{prefix}_meta.npz"), **out)

    stream("probe", probe_parts, n_probe, True)
    stream("inner", inner_parts, n_inner, False)
    rec = {"format": r.FORMATS["prep"], "what": "PREREG_09 1.3 item 1: raw probe / inner arrays (no score)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), **record,
           "probe_rows": n_probe, "inner_rows": n_inner, "sets": rec_sets, "ex_names": FW.ex_names(),
           "storage": "float32 raw: X = [target means, opponent means] (NaN opponent half without an opponent move), "
                      "EX = prep.py extra_features rounded to float32; standardised on use (p09_forward)",
           "files_sha256": {f: r.sha_raw(os.path.join(map_dir, f)) for f in r.PREP_FILES}}
    r.write_json(os.path.join(map_dir, "prep_record.json"), rec)
    return rec


def cmd_prep(a):
    r = R9()
    R = r.R
    c = r.resolve(a, "prep")
    r.blas_env_check(c.preview)
    if os.path.exists(os.path.join(c.map_dir, "prep_record.json")) and not (a.rerun and a.reason):
        raise r.P09Error("prep_record.json exists; rerun only with --rerun --reason TEXT")
    ts = r.tool_state()
    if not c.preview:
        if a.probe or a.limit_chunks:
            raise r.P09Error("real prep takes no --probe / --limit-chunks (the manifest rule decides)")
        r.require_tools(ts, False)
    man, man_sha, _mc = r.manifest_binding(c, False, required=not c.preview)
    if c.preview and a.probe:
        sets = {"P1+P2": ["p1_probe", "p2_probe"], "P1+P2+P3": list(r.PROBE_SETS)}.get(a.probe)
        if sets is None:
            raise r.P09Error("--probe is P1+P2 or P1+P2+P3")
        if "p3_probe" in sets and man is None:
            raise r.P09Error("a P3 preview needs p3_manifest.json")
    else:
        if man is None:
            raise r.P09Error("p3_manifest.json missing (a preview without it needs --probe P1+P2)")
        sets = r.probe_sets_from_manifest(man)
    r.check_pins()
    man08 = R.read_json(r.P08_MANIFEST)
    if r.sha_lf(r.P08_MANIFEST) != r.P08_FROZEN_PINS["results/prereg08/feat/inner_manifest.json"]:
        raise r.P09Error("08 inner_manifest.json differs from its pin")
    probs = R.verify_manifest_chunks(man08, list(R.P1_SET_NAMES) + ["p2_probe"])
    if "p3_probe" in sets:
        probs += r.verify_p3_chunks(man)
    if probs:
        raise r.P09Error(f"chunks differ from the manifests: {probs[:5]}")

    def files_of(manifest_sets, s):
        rows = (manifest_sets.get(s) or {}).get("chunks") or []
        return [(r.path_of(x["file"]), x["sha256"]) for x in rows]

    games = {}
    with open(R.P1_SETS, encoding="utf-8") as f:
        games["p1_probe"] = json.load(f)["probe"]["games"]
    with open(R.P2_SETS, encoding="utf-8") as f:
        games["p2_probe"] = json.load(f)["probe"]["games"]
    if "p3_probe" in sets:
        with open(r.P3_SETS, encoding="utf-8") as f:
            games["p3_probe"] = json.load(f)["probe"]["games"]
    dup08 = man08.get("duplicates") or {}
    probe_parts = []
    for s in sets:
        if s == "p3_probe":
            fl = [(r.path_of(x["file"]), x["sha256"]) for x in man["p3_chunks"]["chunks"]]
            dups = (man.get("duplicates") or {}).get("p3_probe") or []
        else:
            fl = files_of(man08["sets"], s)
            dups = dup08.get(s) or []
        if a.limit_chunks:
            fl = fl[:a.limit_chunks]
        probe_parts.append((s, fl, games[s], dups))
    inner_parts = [(s, files_of(man08["sets"], s)) for s in ("bank", "mono")]
    rec = build_prep(c.map_dir, probe_parts, inner_parts,
                     {**r.mode_flags(c), "probe_used": r.probe_used_label(sets), "probe_sets": sets,
                      "limited": bool(a.limit_chunks), "limit_chunks": a.limit_chunks or None,
                      "p3_manifest_sha256": man_sha,
                      "p08_inner_manifest_sha256": r.sha_lf(r.P08_MANIFEST),
                      "duplicates_excluded_refs": {s: len(d) for s, _f, _g, d in probe_parts},
                      "tool": ts}, log=lambda *x: print(*x, flush=True))
    print(f"prep: probe {rec['probe_used']} {rec['probe_rows']} rows, inner {rec['inner_rows']} rows -> "
          f"{c.map_dir}")
    return 0


# ================================================================ selfcheck (PREREG_09 2.3)
def explore_lp_inner(prep):
    """The exploration's six lp_inner (data/prereg09/tmp/mapping/v_<name>.npz) placed on the prepared inner rows
    (their row order = the exploration's inner_meta.npz ref) -> (rows [6][n, 10], record)."""
    r = R9()
    meta = os.path.join(r.EXPLORE_TMP, "inner_meta.npz")
    with np.load(meta, allow_pickle=False) as z:
        ref_e = np.asarray(z["ref"]).astype(np.int64)
    pos = {int(x): i for i, x in enumerate(prep.inner_ref)}
    if len(ref_e) != len(prep.inner_ref) or any(int(x) not in pos for x in ref_e):
        raise r.P09Error("the exploration's inner rows differ from the prepared inner rows")
    order = np.array([pos[int(x)] for x in ref_e])
    rows, shas = [], {"inner_meta.npz": r.sha_raw(meta)}
    for n in FW.member_names():
        p = os.path.join(r.EXPLORE_TMP, f"v_{n}.npz")
        shas[f"v_{n}.npz"] = r.sha_raw(p)
        with np.load(p, allow_pickle=False) as z:
            lp = np.asarray(z["lp_inner"], dtype=np.float64)
        out = np.zeros_like(lp)
        out[order] = lp
        rows.append(out)
    return rows, {"files_sha256": shas, "same_row_order": bool(np.array_equal(ref_e, prep.inner_ref))}


def cmd_selfcheck(a):
    t0 = time.time()
    r = R9()
    c = r.resolve(a, "selfcheck")
    r.blas_env_check(c.preview)
    if a.threads and not c.preview:
        raise r.P09Error("--threads is preview-only (real runs use 8, PREREG_09 1.3 item 7)")
    venv = r.venv_check(with_torch=True, preview=c.preview)
    rerun = r.guard_outputs(c, a, ["selfcheck.json"])
    ts = r.tool_state()
    if not c.preview:
        r.require_tools(ts, c.precommit)
    cr, cr_sha, cr_commit = r.check_prior(c, ts, c.precommit)
    r.check_inputs_unchanged(c, cr)
    prep = r.Prep(c.map_dir)
    prep_rec = r.prep_binding(c, prep, cr)
    limited = bool(prep.record.get("limited"))
    probs, notes = [], []
    ens_problem = None
    # 1 fusion seam (before torch is imported)
    inner = r.Inner(prep)
    seam = {"Q02H": inner.h["rb"], "Q08": inner.q08["rb"], "Q08_mono": inner.q08["mono_rb"]}
    ex_rows, ex_rec = explore_lp_inner(prep)
    ens_e = FW.lse_mean(ex_rows)
    ens_r = inner.evaluate(ens_e)
    seam.update(ENS=ens_r["rb"], ENS_mono=ens_r["mono_rb"])
    for k in ("Q02H", "Q08", "Q08_mono", "ENS", "ENS_mono"):
        if abs(seam[k] - r.REF[k]) > r.TOL:
            probs.append(f"seam {k} {seam[k]!r} != {r.REF[k]!r}")
    print(f"seam: {seam}", flush=True)
    # 2 retrain on P1+P2 (the exact recipe), numpy forward, torch vs numpy
    folder = os.path.join(c.map_dir, "selfcheck")
    threads = a.threads or r.VENV["threads"]
    mom, recs, info = train_members(prep, ["p1_probe", "p2_probe"], folder, threads,
                                    log=lambda *x: print(*x, flush=True))
    x_equal = None
    with np.load(r.P08_MAP, allow_pickle=False) as z:
        mu8, sd8 = np.asarray(z["mu"]), np.asarray(z["sd"])
    x_equal = {"mu_equal_p1_map": bool(np.array_equal(mom["mu_x"], mu8)),
               "sd_equal_p1_map": bool(np.array_equal(mom["sd_x"], sd8)),
               "max_abs_mu_diff": float(np.max(np.abs(mom["mu_x"] - mu8))),
               "max_abs_sd_diff": float(np.max(np.abs(mom["sd_x"] - sd8)))}
    if not limited and not (x_equal["mu_equal_p1_map"] and x_equal["sd_equal_p1_map"]):
        probs.append(f"P1+P2 mu / sd differ from p1_map.npz: {x_equal}")
    map_tmp = os.path.join(r.TMP, f"p1e_map_selfcheck_{os.getpid()}.npz")
    os.makedirs(r.TMP, exist_ok=True)
    try:
        r.assemble_map(map_tmp, folder, {"what": "selfcheck P1+P2 (record only)"})
        mp = FW.load_map(map_tmp)
        lp_rows, mem_rows = r.inner_map_lp(prep, mp, with_members=True)
        tdiff = torch_numpy_max_diff(map_tmp, prep.inner_X, prep.inner_ex)
    finally:
        if os.path.exists(map_tmp):
            os.remove(map_tmp)
    tdiff["passed"] = bool(tdiff["max_abs_lp_diff"] <= r.TORCH_TOL)
    if not tdiff["passed"]:
        probs.append(f"torch vs numpy forward {tdiff['max_abs_lp_diff']!r} > {r.TORCH_TOL}")
    members = {}
    for spec, lr_, le in zip(FW.MEMBERS, mem_rows, ex_rows):
        rec = recs[spec["name"]]
        members[spec["name"]] = {"cv_logloss": rec["cv"]["logloss"], "e_star": rec["e_star"],
                                 "explore": r.EXPLORE_MEMBERS[spec["name"]],
                                 "cv_logloss_diff": rec["cv"]["logloss"] - r.EXPLORE_MEMBERS[spec["name"]]["cv_logloss"],
                                 "max_abs_lp_inner_diff_vs_explore": float(np.max(np.abs(lr_ - le))),
                                 "logloss_by_epoch": rec["cv"].get("logloss_by_epoch"), "reused": rec.get("reused")}
    ens_new = inner.evaluate(lp_rows)
    ens_diff = ens_new["rb"] - r.REF["ENS"]
    ens_check = {"inner_rank_balanced": ens_new["rb"], "reference": r.REF["ENS"], "difference": ens_diff,
                 "tolerance": r.ENS_TOL, "max_abs_lp_inner_diff_vs_explore": float(np.max(np.abs(lp_rows - ens_e))),
                 "record": inner.short(ens_new, inner.q08)}
    if limited:
        ens_check["passed"] = None
        notes.append("limited preview prep: the 0.002 tolerance is not judged")
    else:
        ens_check["passed"] = bool(abs(ens_diff) <= r.ENS_TOL)
        if not ens_check["passed"]:
            ens_problem = len(probs)
            probs.append(f"retrained ensemble inner {ens_new['rb']!r} differs from {r.REF['ENS']!r} by "
                         f"{ens_diff:+.4f} (> {r.ENS_TOL}): stop and find out why (PREREG_09 2.3 item 2)")
    rep = {"format": r.FORMATS["selfcheck"],
           "what": "PREREG_09 2.3 selfcheck: the tools reproduce the exploration (port check, P1+P2 only; the "
                   "P1+P2 ensemble is record only)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), **r.mode_flags(c), "rerun": rerun,
           "passed": not probs, "problems": probs, "notes": notes, "reference": r.REF, "seam": seam,
           "only_ensemble_tolerance_failed": bool(probs and ens_problem is not None and len(probs) == 1),
           "explore_lp_inner": ex_rec, "x_moments_vs_p1_map": x_equal, "training": info, "members": members,
           "ensemble": ens_check, "torch_vs_numpy": tdiff,
           "weights": {"folder": r.rel(folder) if r.inside(folder, REPO) else folder, "files_sha256": folder_files(folder)},
           "check_report": {"sha256": cr_sha, "commit": cr_commit}, "prep": prep_rec, "hs_mapping": inner.hs_rec,
           "pinned_sha256": r.check_pins(), "tool": ts, "venv": venv, "environment": r.environment_cpu()}
    r.write_json(os.path.join(c.out_dir, "selfcheck.json"), rep)
    print(f"selfcheck: {'passed' if not probs else 'FAILED'}; seam {seam}; ensemble {ens_new['rb']!r} "
          f"({ens_diff:+.4f}); torch vs numpy {tdiff['max_abs_lp_diff']:.2e}")
    if probs:
        for p in probs:
            print("  -", p)
        raise r.Stop("selfcheck failed")
    return 0


# ================================================================ train (PREREG_09 1.3, 1.4)
def _fit_fallback(prep, sets, folder, log=print):
    """Q08P's LR exactly as PREREG_08 1.5 (p1_fuse.fit_mapping_grouped: lambda grid, player-grouped CV, L-BFGS-B
    maxiter 2,000) on the raw float64 X of the used probe rows -> fit file.
    Memory: the pinned fit_mapping_grouped needs float64 X (270k x 1,540 = 3.3 GB) and makes its own standardised
    copy (3.3 GB), transient nan-copies in np.nanmean / np.nanstd and per-fold row copies (~2.7 GB): plan on a peak of
    about 12-14 GB with no other large job running. Not made leaner on purpose: float32 or in-place standardisation
    would change PREREG_08 1.5's numerics (the pinned function standardises internally)."""
    r = R9()
    R = r.R
    PF = R.import_pilot_fuse()
    _L1, F = R.import_p1()
    U = R._import_u1()
    idx = prep.subset(sets)
    X = np.asarray(prep.probe_X[int(idx[0]):int(idx[-1]) + 1], dtype=np.float64)
    if len(X) != len(idx):
        raise r.P09Error("probe rows of the subset are not contiguous")
    y, players = prep.y[idx], prep.players[idx]
    mp = F.fit_mapping_grouped(PF, U, X, y, players, R.LAMBDAS, maxiter=R.MAXITER, log=log)
    del X
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "p1p_fit.npz")
    sets_sha = {"p1": R.DATA_PINS["data/prereg08/p1/sets.json"], "p2": R.DATA_PINS["data/prereg08/p2/sets.json"],
                "p3": r.DATA_PINS["data/prereg09/p3/sets.json"] if "p3_probe" in sets else None}
    np.savez(path + ".tmp.npz", format=np.array(r.FORMATS["fallback_fit"]), W=mp["model"]["W"], b=mp["model"]["b"],
             mu=mp["mu"], sd=mp["sd"], **{"lambda": np.array(mp["lambda"])}, cv=np.array(json.dumps(mp["cv"])),
             refit_loss=np.array(mp["model"]["loss"]), probe_used=np.array(r.probe_used_label(sets)),
             games=np.array(len(idx)), players=np.array(len(set(players.tolist()))),
             sets_sha256=np.array(json.dumps(sets_sha, sort_keys=True)))
    os.replace(path + ".tmp.npz", path)
    return {"lambda": mp["lambda"], "cv": mp["cv"], "refit_loss": mp["model"]["loss"],
            "refit_success": mp["model"]["success"], "refit_nit": mp["model"]["nit"], "games": int(len(idx)),
            "players": int(len(set(players.tolist())))}


def cmd_train(a):
    t0 = time.time()
    r = R9()
    c = r.resolve(a, "train")
    r.blas_env_check(c.preview)
    fb = bool(a.fallback)
    if (a.threads or a.members) and not c.preview:
        raise r.P09Error("--threads / --members are preview-only (real runs: 8 threads, all six members)")
    venv = r.venv_check(with_torch=not fb, preview=c.preview)
    name = "train_fallback_report.json" if fb else "train_report.json"
    p_out = os.path.join(c.out_dir, name)
    if os.path.exists(p_out) and not c.preview and r.git(["ls-files", "--error-unmatch", "--", r.rel(p_out)])[0] == 0:
        raise r.P09Error(f"{name} is committed; never overwritten")
    ts = r.tool_state()
    if not c.preview:
        r.require_tools(ts, c.precommit)
    cr, cr_sha, _cc = r.check_prior(c, ts, c.precommit)
    r.check_inputs_unchanged(c, cr)
    sr, sr_sha, sr_commit = r.prior_report(c, "selfcheck.json", r.FORMATS["selfcheck"], c.precommit)
    override = r.selfcheck_accept(c, sr, getattr(a, "revision", None))
    if (sr.get("check_report") or {}).get("sha256") != cr_sha:
        raise r.P09Error("selfcheck.json was made from another check_report.json")
    diff = r.code_differs(ts, sr.get("tool"), r.TRAINING_CODE)
    if diff:
        raise r.P09Error(f"training code changed since selfcheck: {diff} (selfcheck validated the port)")
    if fb:
        first, _s, _cm = r.prior_report(c, "tune_report.json", r.FORMATS["tune"], False)
        r.fallback_allowed(first)
    prep = r.Prep(c.map_dir)
    prep_rec = r.prep_binding(c, prep, cr)
    sets = cr.get("probe_sets") or ["p1_probe", "p2_probe"]
    if c.preview and prep.record.get("probe_sets"):
        sets = prep.record["probe_sets"]
    if r.probe_used_label(sets) != prep.record.get("probe_used"):
        raise r.P09Error(f"prep probe {prep.record.get('probe_used')} differs from check's {sets}")
    folder = os.path.join(c.map_dir, "fallback" if fb else "train")
    if fb:
        fit = _fit_fallback(prep, sets, folder, log=lambda *x, **k: print(*x, flush=True))
        recs, info, mom = {"lr": fit}, {"probe_used": r.probe_used_label(sets), "rows": fit["games"],
                                        "players": fit["players"]}, None
    else:
        threads = a.threads or r.VENV["threads"]
        members = [x for x in (a.members or "").split(",") if x] or None
        if members and any(m not in FW.member_names() for m in members):
            raise r.P09Error(f"--members: unknown names {members}")
        mom, recs, info = train_members(prep, sets, folder, threads, log=lambda *x: print(*x, flush=True),
                                        members=members)
    complete = fb or all(n in recs for n in FW.member_names())
    rep = {"format": r.FORMATS["train_fallback" if fb else "train"],
           "what": "PREREG_09 1.4 Q08P LR fit (record; judged by tune --fallback)" if fb else
                   "PREREG_09 1.3 ens3fam members on the probe the manifest rule chose (judged by tune)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), **r.mode_flags(c), "complete": complete,
           "probe_used": info["probe_used"], "rows": info["rows"], "players": info["players"],
           "sets_sha256": {"p1": r.R.DATA_PINS["data/prereg08/p1/sets.json"],
                           "p2": r.R.DATA_PINS["data/prereg08/p2/sets.json"],
                           "p3": r.DATA_PINS["data/prereg09/p3/sets.json"] if "p3_probe" in sets else None},
           "members": recs, "training": info, "folder": r.rel(folder) if r.inside(folder, REPO) else folder,
           "files_sha256": folder_files(folder), "prep_record_sha256": prep.record_sha, "prep": prep_rec,
           "check_report_sha256": cr_sha, "selfcheck": {"sha256": sr_sha, "commit": sr_commit},
           "selfcheck_override": override,
           "explore_e_star_record_only": {k: v["e_star"] for k, v in r.EXPLORE_MEMBERS.items()},
           "note": "CPU torch training is not guaranteed bit-reproducible; the weights are what is frozen "
                   "(PREREG_09 1.3 item 7)",
           "tool": ts, "venv": venv, "environment": {**r.environment_cpu(), "torch_threads": info.get("threads")}}
    r.write_json(p_out, rep)
    print(f"train{' --fallback' if fb else ''}: complete {complete}; probe {info['probe_used']} {info['rows']} rows; "
          f"{ {k: v.get('e_star') for k, v in recs.items()} if not fb else recs['lr']['lambda']} -> {folder}")
    return 0


# ================================================================ status
def cmd_status(a):
    r = R9()
    present, expected = p3_presence()
    md = r.MAP_DIR
    out = {"p3_chunks_present": present, "p3_chunks_expected": expected,
           "p3_cutoff": r.P3_CUTOFF.isoformat(), "manifest": os.path.exists(r.MANIFEST),
           "prep_record": os.path.exists(os.path.join(md, "prep_record.json")),
           "selfcheck_members": sorted(f for f in os.listdir(os.path.join(md, "selfcheck"))
                                       if f.startswith("member_")) if os.path.isdir(os.path.join(md, "selfcheck")) else [],
           "train_members": sorted(f for f in os.listdir(os.path.join(md, "train"))
                                   if f.startswith("member_")) if os.path.isdir(os.path.join(md, "train")) else [],
           "reports": {n: os.path.exists(os.path.join(r.OUT, n)) for n in
                       ("check_report.json", "selfcheck.json", "train_report.json", "tune_report.json",
                        "frozen_config.json", "apply_report.json", "apply_audit.json")}}
    print(json.dumps(out, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--precommit", action="store_true")
    m.add_argument("--rerun", action="store_true")
    m.add_argument("--reason", default=None)
    p = sub.add_parser("prep")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--probe", default=None)
    p.add_argument("--limit-chunks", type=int, default=0)
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    for name in ("selfcheck", "train"):
        p = sub.add_parser(name)
        p.add_argument("--precommit", action="store_true")
        p.add_argument("--out-dir", default=None)
        p.add_argument("--threads", type=int, default=0)
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason", default=None)
        if name == "train":
            p.add_argument("--fallback", action="store_true")
            p.add_argument("--members", default=None)
            p.add_argument("--revision", type=int, default=None,
                           help="PREREG_09 rev N (>= 2, committed after selfcheck.json) that accepts a selfcheck whose "
                                "only problem is the 0.002 ensemble tolerance (2.3 item 2)")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    fn = {"manifest": cmd_manifest, "prep": cmd_prep, "selfcheck": cmd_selfcheck, "train": cmd_train,
          "status": cmd_status}[a.cmd]
    try:
        return fn(a)
    except Exception as e:  # noqa: BLE001
        r = sys.modules.get("p09_rank_fuse")
        if r is not None and isinstance(e, r.Stop):
            print(f"STOP: {e}", file=sys.stderr)
            return 1
        if (r is not None and isinstance(e, r.ERRORS)) or type(e).__name__ in ("P1Error", "HSError",
                                                                                 "LocalDataRefused", "P04Error"):
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
