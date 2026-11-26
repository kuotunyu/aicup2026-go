"""prereg09 mapping exploration, step 2 (CPU torch, <= 8 threads): fit candidate mappings of the 1,540 TO features
(optionally + the extra per-game features) to the 10 levels on the 150k fit-probe games only.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg09/explore/mapping/train.py <variant> [...]

For each variant: player-grouped 5-fold CV (p1_fuse.player_folds seed 0, the folds tune used) -> OOF log-probs on the
probe (CV log-loss, top-1, rule-B per-game score; for MLPs the epoch count is chosen by the mean CV log-loss), then a
refit on all 150k probe games -> log-probs of every usable inner bank / mono game. Saved to
data/prereg09/tmp/mapping/v_<variant>.npz and a JSON record. Inner players are never in the training rows
(the probe is drawn from fit players only; inner_manifest.json records 0 move-sequence duplicates).
Data are read from the memory-mapped arrays of prep.py in random minibatches (no full copy in memory).
"""

import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TMP = os.path.join(REPO, "data", "prereg09", "tmp", "mapping")
L = 10
E1 = math.exp(-1)
torch.set_num_threads(int(os.environ.get("MAP_THREADS", "8")))

# variant -> config
V = {
    # controls
    "lr": dict(kind="lin", lam=1e-3, epochs=30, lr=2e-3),
    # ordinal-aware
    "lr_ls10": dict(kind="lin", lam=1e-3, epochs=30, lr=2e-3, smooth=0.10),
    "lr_ls20": dict(kind="lin", lam=1e-3, epochs=30, lr=2e-3, smooth=0.20),
    "lr_rps1": dict(kind="lin", lam=1e-3, epochs=30, lr=2e-3, rps=1.0),
    "lr_ev": dict(kind="lin", lam=1e-3, epochs=30, lr=2e-3, ev=0.05),
    "clm": dict(kind="clm", lam=1e-4, epochs=30, lr=2e-3),
    "lowrank3": dict(kind="lowrank", rank=3, lam=1e-4, epochs=30, lr=2e-3),
    # MLPs (epochs chosen by CV, wd = AdamW decoupled)
    "mlp256": dict(kind="mlp", hidden=[256], drop_in=0.1, drop=0.3, wd=1e-2, epochs=24, lr=1e-3, seeds=[0]),
    "mlp512": dict(kind="mlp", hidden=[512], drop_in=0.2, drop=0.4, wd=1e-2, epochs=24, lr=1e-3, seeds=[0]),
    "mlp512x256": dict(kind="mlp", hidden=[512, 256], drop_in=0.2, drop=0.4, wd=1e-2, epochs=24, lr=1e-3,
                       seeds=[0]),
    "mlp512_ens3": dict(kind="mlp", hidden=[512], drop_in=0.2, drop=0.4, wd=1e-2, epochs=24, lr=1e-3,
                        seeds=[0, 1, 2]),
    "mlp512_rps": dict(kind="mlp", hidden=[512], drop_in=0.2, drop=0.4, wd=1e-2, epochs=24, lr=1e-3, seeds=[0],
                       rps=1.0),
    # extra features
    "clm_l3": dict(kind="clm", lam=1e-3, epochs=30, lr=2e-3),
    "clm_l5": dict(kind="clm", lam=1e-5, epochs=30, lr=2e-3),
    "clm_ex": dict(kind="clm", lam=1e-4, epochs=30, lr=2e-3, ex=True),
    "mlpclm512": dict(kind="mlpclm", hidden=[512], drop_in=0.2, drop=0.4, wd=1e-2, epochs=24, lr=1e-3, seeds=[0]),
    "mlpclm256": dict(kind="mlpclm", hidden=[256], drop_in=0.1, drop=0.3, wd=1e-2, epochs=24, lr=1e-3, seeds=[0]),
    "lowrank2": dict(kind="lowrank", rank=2, lam=1e-4, epochs=30, lr=2e-3),
    "lowrank5": dict(kind="lowrank", rank=5, lam=1e-4, epochs=30, lr=2e-3),
    "lowrank3_l3": dict(kind="lowrank", rank=3, lam=1e-3, epochs=30, lr=2e-3),
    "lowrank3_ex": dict(kind="lowrank", rank=3, lam=1e-4, epochs=30, lr=2e-3, ex=True),
    "mlpclm512_e40": dict(kind="mlpclm", hidden=[512], drop_in=0.2, drop=0.4, wd=1e-2, epochs=40, lr=1e-3,
                          seeds=[0]),
    "mlpclm1024_e40": dict(kind="mlpclm", hidden=[1024], drop_in=0.2, drop=0.5, wd=1e-2, epochs=40, lr=1e-3,
                           seeds=[0]),
    "mlpclm512x128_e40": dict(kind="mlpclm", hidden=[512, 128], drop_in=0.2, drop=0.3, wd=1e-2, epochs=40, lr=1e-3,
                              seeds=[0]),
    "mlpclm512_e40_ens3": dict(kind="mlpclm", hidden=[512], drop_in=0.2, drop=0.4, wd=1e-2, epochs=40, lr=1e-3,
                               seeds=[0, 1, 2]),
    "mlpclm512_e40_ex": dict(kind="mlpclm", hidden=[512], drop_in=0.2, drop=0.4, wd=1e-2, epochs=40, lr=1e-3,
                             seeds=[0], ex=True),
    "mlpclm256_s1": dict(kind="mlpclm", hidden=[256], drop_in=0.1, drop=0.3, wd=1e-2, epochs=24, lr=1e-3, seeds=[1]),
    "mlpclm256_s2": dict(kind="mlpclm", hidden=[256], drop_in=0.1, drop=0.3, wd=1e-2, epochs=24, lr=1e-3, seeds=[2]),
    "mlpclm256_ex": dict(kind="mlpclm", hidden=[256], drop_in=0.1, drop=0.3, wd=1e-2, epochs=24, lr=1e-3, seeds=[0],
                         ex=True),
    "mlp512_s1": dict(kind="mlp", hidden=[512], drop_in=0.2, drop=0.4, wd=1e-2, epochs=24, lr=1e-3, seeds=[1]),
    "lr_ex": dict(kind="lin", lam=1e-3, epochs=30, lr=2e-3, ex=True),
    "mlp512_ex": dict(kind="mlp", hidden=[512], drop_in=0.2, drop=0.4, wd=1e-2, epochs=24, lr=1e-3, seeds=[0],
                      ex=True),
}


def load():
    X = np.load(os.path.join(TMP, "probe_X.npy"), mmap_mode="r")
    XI = np.load(os.path.join(TMP, "inner_X.npy"), mmap_mode="r")
    m = np.load(os.path.join(TMP, "probe_meta.npz"))
    ex = np.load(os.path.join(TMP, "probe_ex.npy")).astype(np.float64)
    exi = np.load(os.path.join(TMP, "inner_ex.npy")).astype(np.float64)
    mu, sd = np.nanmean(ex, 0), np.nanstd(ex, 0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    ex = np.nan_to_num((ex - mu) / sd).astype(np.float32)
    exi = np.nan_to_num((exi - mu) / sd).astype(np.float32)
    return X, XI, m["y"], m["fold"], ex, exi


class Lin(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = torch.nn.Linear(d, L)
        torch.nn.init.zeros_(self.f.weight)
        torch.nn.init.zeros_(self.f.bias)

    def forward(self, x):
        return torch.log_softmax(self.f(x), 1)

    def penalty(self):
        return (self.f.weight ** 2).sum()


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


class CLM(torch.nn.Module):
    """proportional-odds cumulative logit: P(y <= k) = sigmoid(theta_k - w.x), theta increasing."""

    def __init__(self, d):
        super().__init__()
        self.w = torch.nn.Linear(d, 1, bias=False)
        torch.nn.init.zeros_(self.w.weight)
        self.t0 = torch.nn.Parameter(torch.tensor(-4.0))
        self.dt = torch.nn.Parameter(torch.full((L - 2,), math.log(math.e - 1)))   # softplus -> 1

    def forward(self, x):
        f = self.w(x)                                                               # [B,1]
        th = torch.cat([self.t0.view(1), self.t0 + torch.cumsum(torch.nn.functional.softplus(self.dt), 0)])
        c = torch.sigmoid(th.view(1, -1) - f)                                       # [B,9]
        c = torch.cat([torch.zeros_like(f), c, torch.ones_like(f)], 1)
        p = (c[:, 1:] - c[:, :-1]).clamp_min(1e-12)
        return torch.log(p)

    def penalty(self):
        return (self.w.weight ** 2).sum()


def clm_logp(f, t0, dt):
    th = torch.cat([t0.view(1), t0 + torch.cumsum(torch.nn.functional.softplus(dt), 0)])
    c = torch.sigmoid(th.view(1, -1) - f)
    c = torch.cat([torch.zeros_like(f), c, torch.ones_like(f)], 1)
    return torch.log((c[:, 1:] - c[:, :-1]).clamp_min(1e-12))


class MLPCLM(torch.nn.Module):
    """MLP -> one scalar strength -> the cumulative-link head (ordinal MLP)."""

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


def make(cfg, d, seed):
    torch.manual_seed(seed)
    k = cfg["kind"]
    if k == "lin":
        return Lin(d)
    if k == "lowrank":
        return LowRank(d, cfg["rank"])
    if k == "clm":
        return CLM(d)
    if k == "mlpclm":
        return MLPCLM(d, cfg["hidden"], cfg["drop_in"], cfg["drop"])
    return MLP(d, cfg["hidden"], cfg["drop_in"], cfg["drop"])


def soft_targets(y, eps):
    T = np.zeros((len(y), L), dtype=np.float32)
    T[np.arange(len(y)), y] = 1.0 - eps
    for i, k in enumerate(y):
        nb = [j for j in (k - 1, k + 1) if 0 <= j < L]
        for j in nb:
            T[i, j] += eps / len(nb)
    return T


def loss_fn(cfg, lp, yb, Tb):
    ll = -(Tb * lp).sum(1).mean() if Tb is not None else torch.nn.functional.nll_loss(lp, yb)
    if cfg.get("rps"):
        p = lp.exp()
        cdf = torch.cumsum(p, 1)[:, :-1]
        tgt = (torch.arange(L - 1).view(1, -1) >= yb.view(-1, 1)).float()
        ll = ll + cfg["rps"] * ((cdf - tgt) ** 2).sum(1).mean()
    if cfg.get("ev"):
        p = lp.exp()
        ev = (p * torch.arange(L, dtype=p.dtype)).sum(1)
        ll = ll + cfg["ev"] * ((ev - yb.float()) ** 2).mean()
    return ll


def batches(rows, X, ex, bs, rng, shuffle=True):
    rows = rng.permutation(rows) if shuffle else rows
    for s in range(0, len(rows), bs):
        r = np.sort(rows[s:s + bs])
        xb = np.asarray(X[r], dtype=np.float32)
        if ex is not None:
            xb = np.concatenate([xb, ex[r]], 1)
        yield r, torch.from_numpy(xb)


def predict(model, X, ex, rows=None, bs=8192):
    model.eval()
    rows = np.arange(X.shape[0]) if rows is None else rows
    out = np.zeros((len(rows), L))
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            r = rows[s:s + bs]
            xb = np.asarray(X[r], dtype=np.float32)
            if ex is not None:
                xb = np.concatenate([xb, ex[r]], 1)
            out[s:s + len(r)] = model(torch.from_numpy(xb)).double().numpy()
    return out


def rule_b_score(lp, y):
    p = np.exp(lp)
    left = np.zeros_like(p)
    right = np.zeros_like(p)
    left[:, 1:] = p[:, :-1]
    right[:, :-1] = p[:, 1:]
    pred = np.argmax(p + E1 * (left + right), 1)
    d = np.abs(pred - y)
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))


def train(cfg, X, ex, y, rows, seed, epochs, T=None, stop_after=None, eval_rows=None):
    """OneCycle schedule over `epochs`; stops after `stop_after` epochs (default all). With eval_rows: returns the
    per-epoch probabilities of those rows (MLP early stopping by CV)."""
    d = X.shape[1] + (ex.shape[1] if ex is not None else 0)
    model = make(cfg, d, seed)
    if cfg["kind"] in ("mlp", "mlpclm"):
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    bs = 512
    steps = epochs * math.ceil(len(rows) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    yt = torch.from_numpy(y)
    Tt = torch.from_numpy(T) if T is not None else None
    snaps = []
    for ep in range(stop_after or epochs):
        model.train()
        for r, xb in batches(rows, X, ex, bs, rng):
            lp = model(xb)
            loss = loss_fn(cfg, lp, yt[r], Tt[r] if Tt is not None else None)
            if cfg["kind"] not in ("mlp", "mlpclm"):
                loss = loss + cfg["lam"] * model.penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if eval_rows is not None:
            snaps.append(np.exp(predict(model, X, ex, eval_rows)).astype(np.float32))
    return model, snaps


def player_subset(frac, seed=1):
    """whole players, seeded (as p1_explore.py E2): the first round(frac x players) of a permutation."""
    pl = np.load(os.path.join(TMP, "probe_meta.npz"))["players"]
    uniq = np.unique(pl)
    order = np.random.default_rng(seed).permutation(uniq)
    keep = set(order[:int(round(frac * len(order)))].tolist())
    return np.array([p in keep for p in pl])


def run(name):
    base, _, frac = name.partition("@")      # 'mlpclm256@0.2': learning-curve run on 20 % of the probe players
    cfg = V[base]
    t0 = time.time()
    X, XI, y, fold, ex, exi = load()
    if frac:
        fold = np.where(player_subset(float(frac)), fold, -1)     # rows outside the subset: never used
    use_ex = cfg.get("ex", False)
    exp_, exi_ = (ex, exi) if use_ex else (None, None)
    T = soft_targets(y, cfg["smooth"]) if cfg.get("smooth") else None
    seeds = cfg.get("seeds", [0])
    epochs = cfg["epochs"]
    sub = np.nonzero(fold >= 0)[0]
    n = len(y)
    is_mlp = cfg["kind"] in ("mlp", "mlpclm")
    E = epochs if is_mlp else 1
    oof_p = np.zeros((E, n, L), dtype=np.float32)       # OOF probabilities (seed-averaged) per epoch
    for f in range(5):
        tr, te = np.nonzero((fold != f) & (fold >= 0))[0], np.nonzero(fold == f)[0]
        for s in seeds:
            if is_mlp:
                _m, snaps = train(cfg, X, exp_, y, tr, s, epochs, eval_rows=te)
                for e in range(E):
                    oof_p[e, te] += snaps[e] / len(seeds)
            else:
                model, _ = train(cfg, X, exp_, y, tr, s, epochs, T=T)
                oof_p[0, te] += (np.exp(predict(model, X, exp_, te)) / len(seeds)).astype(np.float32)
        lt = np.log(np.clip(oof_p[-1, te].astype(np.float64), 1e-300, None))
        print(f"{name} fold {f}: CV ll (last epoch) {-lt[np.arange(len(te)), y[te]].mean():.4f} "
              f"{time.time() - t0:.0f}s", flush=True)
    ll_ep = [float(-np.log(np.clip(oof_p[e][sub, y[sub]].astype(np.float64), 1e-300, None)).mean())
             for e in range(E)]
    best_ep = int(np.argmin(ll_ep)) + 1 if is_mlp else epochs
    oof = np.log(np.clip(oof_p[best_ep - 1 if is_mlp else 0].astype(np.float64), 1e-300, None))
    del oof_p
    cv = {"logloss": float(-oof[sub, y[sub]].mean()), "top1": float((oof[sub].argmax(1) == y[sub]).mean()),
          "rule_b_score_per_game": float(rule_b_score(oof[sub], y[sub]).mean()), "games": int(len(sub)),
          "logloss_by_epoch": ll_ep if is_mlp else None, "epochs_chosen": best_ep, "epochs_schedule": epochs}
    print(f"{name}: CV {cv['logloss']:.4f} top1 {cv['top1']:.4f} ruleB {cv['rule_b_score_per_game']:.4f} "
          f"epochs {best_ep}/{epochs}", flush=True)
    # refit on all probe games (same schedule, stopped at the CV-chosen epoch)
    Pi = np.zeros((XI.shape[0], L))
    for s in seeds:
        model, _ = train(cfg, X, exp_, y, sub, s, epochs, T=T, stop_after=best_ep if is_mlp else None)
        Pi += np.exp(predict(model, XI, exi_)) / len(seeds)
    lp_inner = np.log(np.clip(Pi, 1e-300, None))
    np.savez(os.path.join(TMP, f"v_{name}.npz"), oof=oof.astype(np.float32), lp_inner=lp_inner)
    rec = {"variant": name, "config": cfg, "cv": cv, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(TMP, f"v_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        run(nm)
