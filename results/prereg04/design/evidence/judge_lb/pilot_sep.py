"""Judge probe (inner only): can a pilot threshold on the nested increment separate a useful new source from a useless one?

Base combos: Q01 = R20 + O_px, Q02 = R20 + O_px + SWA + SWB.  Add-ons of known quality:
  useless-ish : pure noise (proxy_null), target-side proxy T_px, C6A/C6B correct colour (A0/B0), C6A d4 combo
  useful      : SWA / SWB / SWA+SWB (on Q01 only)
  weakened    : O_px from one game phase only; O_px with added per-game noise (half / quarter SNR) -> on R20
Also the pairwise overlap between O_px and SW (O on top of R20+SW).
Read-only on the repo; inner-side files only; no validation dump, no answer file, no score.py; CPU.
"""
import json
import os
import sys
import time
import types

sys.dont_write_bytecode = True
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
P04 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BPROXY = os.path.join(P04, "B_proxy")
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, BPROXY)
import numpy as np  # noqa: E402

import p02_rank_fuse as rf  # noqa: E402
import p03_rank_fuse as p3  # noqa: E402
import fuse as fz  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    t0 = time.time()
    c = types.SimpleNamespace(p02_dumps=os.path.join(p3.P02_RANK, "dumps"), p02_local=p3.P02_RANK,
                              split=p3.SPLIT_CSV, smoke=False)
    inner = p3.prepare_inner(c)
    keys, Xf, Xe, Sf, Se = inner["keys"], inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"]
    boot = inner["boot"]
    n = keys["n"]
    logits = {p3.C5_DUMP: p3.load_c5_inner(c, n)["logits"]}
    swap, s0 = {}, {}
    for arm in ("a", "b"):
        ddir = os.path.join(p3.P03_ROOT, f"rank_{arm}", "dumps")
        d = rf.load_dump(os.path.join(ddir, f"inner_p03{arm}_d4.npz"), n, 8)
        logits[f"p03{arm}_d4"] = d["logits"]
        s0[arm] = rf.log_softmax(d["logits"][:, 0, :].astype(np.float64))
        swap[arm] = rf.log_softmax(p3.load_swap(os.path.join(ddir, f"inner_p03{arm}_s0_swap.npz"), n)["logits"].astype(np.float64))
    lp00 = p3.combo_logprobs(dict(p3.COMBOS)["P00"], logits)
    lpA = p3.combo_logprobs(("C6A",), logits)

    def load_px(fname):
        z = np.load(os.path.join(BPROXY, fname))
        row = {(r, g): i for i, (r, g) in enumerate(zip(z["rank"].tolist(), z["game_id"].tolist()))}
        idx = np.array([row[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])])
        assert all(z["player_id"][idx] == np.array(keys["player_id"]))
        return [str(v) for v in z["variants"]], z["ll"][idx]
    var, ll = load_px("proxy_inner.npz")
    _, lln = load_px("proxy_null.npz")
    io, it = var.index("o_choice_a20"), var.index("t_choice_a20")
    llo = ll[:, io].sum(1)                     # [n,10] opponent-side per-game LL sum
    O = rf.log_softmax(llo)
    T = rf.log_softmax(ll[:, it].sum(1))
    NUL = rf.log_softmax(lln[:, io].sum(1))
    rng = np.random.default_rng(12345)
    cen = llo - llo.mean(1, keepdims=True)
    sd = float(np.std(cen))
    O_half = rf.log_softmax(llo + rng.normal(0, sd, llo.shape))          # signal var / noise var ~ 1
    O_quarter = rf.log_softmax(llo + rng.normal(0, np.sqrt(3) * sd, llo.shape))
    O_ph = {nm: rf.log_softmax(ll[:, io, k]) for k, nm in enumerate(("open", "mid", "late"))}

    src = {"O": O, "T": T, "NULL": NUL, "SWA": swap["a"], "SWB": swap["b"], "A0": s0["a"], "B0": s0["b"],
           "C6A": lpA, "O_half": O_half, "O_quarter": O_quarter, **{f"O_{k}": v for k, v in O_ph.items()}}
    pool = lambda S, lp: rf.pooled(S, lp, "log")  # noqa: E731
    mf00, me00 = pool(Sf, lp00), pool(Se, lp00)
    Ef = {k: pool(Sf, v) for k, v in src.items()}
    Ee = {k: pool(Se, v) for k, v in src.items()}
    opf, ope = Xf.mop, Xe.mop

    cache = {}

    def run(terms):
        key = tuple(terms)
        if key not in cache:
            P, ws = fz.gen_oof(Xf, Xe, mf00, me00, [opf] + [Ef[t] for t in terms], [ope] + [Ee[t] for t in terms])
            m, sc = rf.metrics(rf.rule_b_np(P), Xe.y, Xe.n, Xe.player)
            cache[key] = (m["rank_balanced"], m["question_mean"], sc, ws)
        return cache[key]

    rb0 = run([])[0]
    assert abs(rb0 - 0.5854702521055817) < 1e-12, rb0
    out = {"R20": rb0, "rows": []}

    def nested(base, add, label):
        rb_b, qm_b, sc_b, _ = run(base)
        rb_n, qm_n, sc_n, ws = run(base + add)
        d = boot.stats(sc_n) - boot.stats(sc_b)
        wadd = [w[len(base) + 1:] for w in ws]
        rec = {"label": label, "base": "+".join(base) or "R20", "add": "+".join(add), "rb_base": rb_b, "rb_new": rb_n,
               "delta": rb_n - rb_b, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
               "se_boot": float(np.std(d, ddof=1)), "dqm": float(np.mean(sc_n - sc_b)),
               "weights_add_by_fold": [[round(x, 3) for x in w] for w in wadd],
               "all_folds_positive": bool(all(min(w) > 1e-6 for w in wadd))}
        out["rows"].append(rec)
        print(f"{label:10s} {rec['base']:14s} + {rec['add']:12s} d {rec['delta']:+.4f} [{rec['ci95'][0]:+.4f},{rec['ci95'][1]:+.4f}]"
              f" se {rec['se_boot']:.4f} dqm {rec['dqm']:+.4f} allpos {rec['all_folds_positive']} w {rec['weights_add_by_fold']}"
              f" ({time.time() - t0:.0f}s)", flush=True)

    Q01, Q02 = ["O"], ["O", "SWA", "SWB"]
    for add, lab in ((["NULL"], "useless"), (["T"], "useless"), (["A0"], "useless"), (["B0"], "useless"),
                     (["C6A"], "useless"), (["SWA"], "useful"), (["SWB"], "useful"), (["SWA", "SWB"], "useful")):
        nested(Q01, add, lab)
    for add, lab in ((["NULL"], "useless"), (["T"], "useless"), (["A0"], "useless"), (["C6A"], "useless")):
        nested(Q02, add, lab)
    nested(["SWA", "SWB"], ["O"], "overlap")
    for add in (["O_half"], ["O_quarter"], ["O_open"], ["O_mid"], ["O_late"]):
        nested([], add, "weakened")
    for add in (["O_half"], ["O_quarter"]):
        nested(["SWA", "SWB"], add, "weak/SW")
    out["seconds"] = time.time() - t0
    with open(os.path.join(HERE, "pilot_sep.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
