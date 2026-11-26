"""p04 design probe (inner only): is the opponent-side signal already available from the committed prereg03
colour-swapped whole-game logits (inner_p03{a,b}_s0_swap.npz)? Compared with the CPU proxy opponent term O_px.

Read-only on the repo; reads only inner-side dumps (results/prereg02/rank/dumps/inner_*, pseudo_inner.json,
opening_inner.npz, results/prereg03/rank_{a,b}/dumps/inner_*); no validation dump, no answer file, no score.py.
Reuses task B's generalised calibration (gen_oof) unchanged.
"""
import json
import os
import sys
import time
import types

sys.dont_write_bytecode = True
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
BPROXY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "B_proxy")
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, BPROXY)
import numpy as np  # noqa: E402

import p02_rank_fuse as rf  # noqa: E402
import p03_rank_fuse as p3  # noqa: E402
import fuse as fz  # noqa: E402  (task B's script; main() not called)
from c10data import LEVELS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
L = 10


def main():
    t0 = time.time()
    c = types.SimpleNamespace(p02_dumps=os.path.join(p3.P02_RANK, "dumps"), p02_local=p3.P02_RANK,
                              split=p3.SPLIT_CSV, smoke=False)
    inner = p3.prepare_inner(c)
    keys, Xf, Xe, Sf, Se = inner["keys"], inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"]
    boot = inner["boot"]
    n = keys["n"]
    logits = {p3.C5_DUMP: p3.load_c5_inner(c, n)["logits"]}
    swap = {}
    s0 = {}
    for arm in ("a", "b"):
        ddir = os.path.join(p3.P03_ROOT, f"rank_{arm}", "dumps")
        diff = p3.key_array_diff(os.path.join(ddir, "inner_keys.npz"), os.path.join(c.p02_dumps, "inner_keys.npz"),
                                 ("rank", "game_id", "player_id", "color"))
        if diff:
            raise SystemExit(f"arm {arm} inner keys differ: {diff}")
        d = rf.load_dump(os.path.join(ddir, f"inner_p03{arm}_d4.npz"), n, 8)
        logits[f"p03{arm}_d4"] = d["logits"]
        s0[arm] = rf.log_softmax(d["logits"][:, 0, :].astype(np.float64))
        sw = p3.load_swap(os.path.join(ddir, f"inner_p03{arm}_s0_swap.npz"), n)
        swap[arm] = rf.log_softmax(sw["logits"].astype(np.float64))
    lp00 = p3.combo_logprobs(dict(p3.COMBOS)["P00"], logits)
    lpA = p3.combo_logprobs(("C6A",), logits)
    lpB = p3.combo_logprobs(("C6B",), logits)

    z = np.load(os.path.join(BPROXY, "proxy_inner.npz"))
    row = {(r, g): i for i, (r, g) in enumerate(zip(z["rank"].tolist(), z["game_id"].tolist()))}
    idx = np.array([row[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])])
    if not all(z["player_id"][idx] == np.array(keys["player_id"])):
        raise SystemExit("proxy player ids differ")
    variants = [str(v) for v in z["variants"]]
    llv = z["ll"][idx]
    pxo = rf.log_softmax(llv[:, variants.index("o_choice_a20")].sum(1))

    lab = keys["label"]
    scored = z["scored"][idx] if "scored" in z.files else np.ones(len(idx), dtype=bool)
    acc = {"C6A_s0": float(np.mean(s0["a"][scored].argmax(1) == lab[scored])),
           "C6A_swap": float(np.mean(swap["a"][scored].argmax(1) == lab[scored])),
           "C6B_s0": float(np.mean(s0["b"][scored].argmax(1) == lab[scored])),
           "C6B_swap": float(np.mean(swap["b"][scored].argmax(1) == lab[scored])),
           "O_px": float(np.mean(pxo[scored].argmax(1) == lab[scored]))}
    print("per-game accuracy on bank games:", {k: round(v * 100, 2) for k, v in acc.items()}, flush=True)

    pool = lambda S, lp: rf.pooled(S, lp, "log")  # noqa: E731
    mf00, me00 = pool(Sf, lp00), pool(Se, lp00)
    src = {"O_px": pxo, "SWA": swap["a"], "SWB": swap["b"], "C6A": lpA, "C6B": lpB, "A0": s0["a"], "B0": s0["b"]}
    Ef = {k: pool(Sf, v) for k, v in src.items()}
    Ee = {k: pool(Se, v) for k, v in src.items()}
    opf, ope = Xf.mop, Xe.mop

    results = {}
    sc_ref = None

    def record(name, P, ws=None):
        nonlocal sc_ref
        pred = rf.rule_b_np(P)
        m, sc = rf.metrics(pred, Xe.y, Xe.n, Xe.player)
        rec = {"rank_balanced": m["rank_balanced"], "question_mean": m["question_mean"], "per_level": m["per_level"],
               "weights_by_fold": ws}
        if name == "R20":
            sc_ref = sc
        else:
            d = boot.stats(sc) - boot.stats(sc_ref)
            rec["diff_vs_R20"] = m["rank_balanced"] - results["R20"]["rank_balanced"]
            rec["diff_vs_R20_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            rec["question_mean_diff_vs_R20"] = float(np.mean(sc - sc_ref))
        results[name] = rec
        print(f"{name:40s} rb {m['rank_balanced']:.4f} qm {m['question_mean']:.4f}"
              + (f"  d {rec['diff_vs_R20']:+.4f} [{rec['diff_vs_R20_ci95'][0]:+.4f}, {rec['diff_vs_R20_ci95'][1]:+.4f}]"
                 f" dqm {rec['question_mean_diff_vs_R20']:+.4f}" if name != "R20" else "")
              + (f"  w0 {np.round(ws[0], 3).tolist()}" if ws else "") + f" ({time.time() - t0:.0f}s)", flush=True)

    P, ws = fz.gen_oof(Xf, Xe, mf00, me00, [opf], [ope])
    record("R20", P)
    if abs(results["R20"]["rank_balanced"] - 0.5854702521055817) > 1e-12:
        raise SystemExit("R20 seam failed")

    def run(name, terms):
        P, ws = fz.gen_oof(Xf, Xe, mf00, me00, [opf] + [Ef[t] for t in terms], [ope] + [Ee[t] for t in terms])
        record(name, P, ws)

    run("R20 + O_px", ["O_px"])
    run("R20 + SWA (C6A colour-swapped, s0)", ["SWA"])
    run("R20 + SWB (C6B colour-swapped, s0)", ["SWB"])
    run("R20 + SWA + SWB", ["SWA", "SWB"])
    run("R20 + A0 (C6A correct colour, s0)", ["A0"])
    run("R20 + A0 + SWA", ["A0", "SWA"])
    run("R20 + O_px + SWA", ["O_px", "SWA"])
    run("R20 + O_px + SWA + SWB", ["O_px", "SWA", "SWB"])
    run("R20 + C6A + SWA", ["C6A", "SWA"])
    run("R20 + O_px + C6A + SWA", ["O_px", "C6A", "SWA"])
    for nm in ("SWA", "O_px"):
        P, ws = fz.gen_oof(Xf, Xe, Ef[nm], Ee[nm], [], [])
        record(f"{nm} alone", P, ws)
    # per-game correlation of level slopes on bank games
    lvl = np.arange(L) - 4.5
    slope = lambda u: ((u - u.mean(1, keepdims=True)) @ lvl) / (lvl @ lvl)  # noqa: E731
    sb = scored
    corr = {f"{a} vs {b}": float(np.corrcoef(slope(src2[sb]), slope(src3[sb]))[0, 1])
            for a, src2 in (("C5", lp00), ("A0", s0["a"])) for b, src3 in (("SWA", swap["a"]), ("O_px", pxo), ("A0", s0["a"]))}
    corr["SWA vs O_px"] = float(np.corrcoef(slope(swap["a"][sb]), slope(pxo[sb]))[0, 1])
    print("per-game slope correlations (bank games):", json.dumps({k: round(v, 3) for k, v in corr.items()}), flush=True)
    with open(os.path.join(HERE, "swap_probe.json"), "w", encoding="utf-8") as fh:
        json.dump({"per_game_accuracy_bank": acc, "combinations": results, "slope_corr_bank": corr,
                   "seconds": time.time() - t0}, fh, indent=1)


if __name__ == "__main__":
    main()
