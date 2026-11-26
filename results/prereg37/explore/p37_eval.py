"""prereg37 explore: checkpoint / weight selection on the D pools, the one judgement against M36 on the E pools,
apply (PLAN.md 3-5). The pools and the G terms are those of prereg36 (data/prereg36/terms_D|E); only the
embeddings differ.

    PY results/prereg37/explore/p37_eval.py select      # -> tune.json (commit it before judge)
    PY results/prereg37/explore/p37_eval.py judge       # -> eval.json (once)
    PY results/prereg37/explore/p37_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg37/explore/p37_eval.py assemble --v1 <v1 submission> --src <official tests dir>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P36 = os.path.join(REPO, "results", "prereg36", "explore")
sys.path.insert(0, P36)
import p36_eval as EV  # noqa: E402   (TERMS_DIR = data/prereg36/terms_*; adds its own sys.path entries)
import p36_lib as L  # noqa: E402
import p06_sim as sim  # noqa: E402
import p06_style as st  # noqa: E402

EMB37 = os.path.join(REPO, "data", "prereg37", "emb")
EMB36 = L.EMB
WE_GRID = (0.5, 1, 2, 4, 8, 16)
BOOT_SEED = 37
ADOPT_MEAN = 0.01
TOTAL_STEPS = 160000
P36_TUNE = os.path.join(P36, "tune.json")
P36_EVAL = os.path.join(P36, "eval.json")
OUT_APPLY = os.path.join(REPO, "results", "prereg37", "apply")


def emb37(stem, name):
    return os.path.join(EMB37, stem, name + ".npz")


def cmd_select(a):
    with open(a.train_record, encoding="utf-8") as f:
        tr = json.load(f)
    if not tr.get("finished") or tr.get("smoke") or tr["steps"] != TOTAL_STEPS:
        raise SystemExit("train_record.json is not the finished 160,000-step run")
    pools = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    cnn = {}
    for name in tr["regular_checkpoints"]:
        stem = os.path.splitext(name)[0]
        f = emb37(stem, "store_D")
        if not os.path.exists(f):
            raise SystemExit("missing %s" % f)
        e = EV.Emb.load(f)
        if e.sha != tr["checkpoints"][name]:
            raise SystemExit("%s was not made from the recorded checkpoint %s" % (f, name))
        cnn[stem] = float(np.mean([st.score(st.csls(EV.e36(e, p)[0], EV.CNN_K), p["owner"]) for p in pools]))
    best = max(sorted(cnn), key=lambda s: cnn[s])          # ties: the earliest checkpoint
    e = EV.Emb.load(emb37(best, "store_D"))
    embs = [EV.e36(e, p) for p in pools]
    EV.WE_GRID = WE_GRID
    rows = EV.tune_grid(pools, embs)
    top = max(rows, key=lambda r: r[0])                    # first maximum in grid order
    w, kG = EV.g_combo()
    with open(P36_TUNE, encoding="utf-8") as f:
        t36 = json.load(f)
    key = "cnn_only_csls%d_by_checkpoint" % EV.CNN_K
    rec = {"format": "c10-prereg37 tune v1", "d_seeds": EV.D_SEEDS, key: cnn,
           "checkpoint": {"name": best + ".pt", "sha256": e.sha, "cnn_only_mean": cnn[best]},
           "G": {"weights_O_L_JO_S2_PA": list(w), "csls_k": kG},
           "grid": [{"mean": r[0], "wE": r[1], "k": r[2]} for r in rows],
           "M": {"wE": top[1], "csls_k": top[2], "tune_mean": top[0]},
           "M36_on_D": {"tune_mean": t36["M"]["tune_mean"], "cnn_only_mean": t36["checkpoint"]["cnn_only_mean"]},
           "train_record_sha256": L.sha256_file(a.train_record)}
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in (key, "checkpoint", "M", "M36_on_D")}, indent=1))
    return 0


def cmd_judge(a):
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    if not L.git_committed(a.tune):
        raise SystemExit("tune.json is not committed (or differs from HEAD); commit it before the judgement")
    with open(a.tune, encoding="utf-8") as f:
        tune = json.load(f)
    with open(P36_TUNE, encoding="utf-8") as f:
        t36 = json.load(f)
    with open(P36_EVAL, encoding="utf-8") as f:
        rec36 = json.load(f)
    store = sim.Store(L.FEATS, load=())
    stem = os.path.splitext(tune["checkpoint"]["name"])[0]
    e37 = EV.Emb.load(emb37(stem, "store_E"), colour=store.color)
    if e37.sha != tune["checkpoint"]["sha256"]:
        raise SystemExit("store_E embeddings were not made from the chosen checkpoint")
    stem36 = os.path.splitext(t36["checkpoint"]["name"])[0]
    e36 = EV.Emb.load(os.path.join(EMB36, stem36, "store_E.npz"), colour=store.color)
    if e36.sha != t36["checkpoint"]["sha256"]:
        raise SystemExit("prereg36 store_E embeddings do not match its tune.json")
    pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    w, _kG = EV.g_combo()

    def run(e, wE, k):
        tot, byk, cnn = [], [], []
        for p in pools:
            E = EV.e36(e, p)
            S = st.csls(EV.fused(p, E, w, wE), k)
            tot.append(st.score(S, p["owner"]))
            byk.append(EV.score_by_k(S, p))
            cnn.append(st.score(st.csls(E[0], EV.CNN_K), p["owner"]))
        return np.array(tot), EV.mean_by_k(byk), float(np.mean(cnn))

    m36, k36, c36 = run(e36, t36["M"]["wE"], t36["M"]["csls_k"])
    if float(m36.mean()) != rec36["means"]["M"] or m36.tolist() != rec36["per_pool"]["M"]:
        raise SystemExit("seam: M36 on the E pools %r != prereg36 eval.json %r"
                         % (float(m36.mean()), rec36["means"]["M"]))
    m37, k37, c37 = run(e37, tune["M"]["wE"], tune["M"]["csls_k"])
    d = m37 - m36
    lo, hi = EV.bootstrap(d, seed=BOOT_SEED)
    pos = int((d > 0).sum())
    col = {name: float(np.mean([st.score(st.csls(EV.e36_colour(e37, p, cross)[0], EV.CNN_K), p["owner"])
                                for p in pools])) for name, cross in (("same_colour", False), ("cross_colour", True))}
    rec = {"format": "c10-prereg37 eval v1", "e_seeds": EV.E_SEEDS, "checkpoint": tune["checkpoint"],
           "M": tune["M"], "M36": {"checkpoint": t36["checkpoint"], "M": t36["M"]},
           "tune_sha256": L.sha256_file(a.tune), "git_head": L.git_head(), "seam_M36_identical": True,
           "means": {"M37": float(m37.mean()), "M36": float(m36.mean()), "E37_only_csls%d" % EV.CNN_K: c37,
                     "E36_only_csls%d" % EV.CNN_K: c36},
           "per_pool": {"M37": m37.tolist(), "M36": m36.tolist()},
           "M37_minus_M36": {"mean": float(d.mean()), "ci95": [lo, hi], "pools_positive": pos},
           "by_k": {"M37": k37, "M36": k36},
           "E37_only_by_colour_csls%d" % EV.CNN_K: col,
           "rule": "adopt M37 if mean(M37 - M36) >= %.2f and ci95 low > 0 and M37 > M36 in >= 18 of 20 pools "
                   "(pool bootstrap, %d samples, seed %d); second use of the E pools"
                   % (ADOPT_MEAN, EV.BOOT, BOOT_SEED),
           "adopt": bool(d.mean() >= ADOPT_MEAN and lo > 0 and pos >= int(np.ceil(0.9 * len(d))))}
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("means", "M37_minus_M36", "by_k", "adopt")}, indent=1))
    return 0


def cmd_apply(a):
    EV.emb_path = emb37
    return EV.cmd_apply(a)


def cmd_assemble(a):
    return EV.cmd_assemble(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--train-record", default=os.path.join(HERE, "train_record.json"))
    s.add_argument("--out", default=os.path.join(HERE, "tune.json"))
    j = sub.add_parser("judge")
    j.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    j.add_argument("--out", default=os.path.join(HERE, "eval.json"))
    q = sub.add_parser("apply")
    q.add_argument("--src", required=True)
    q.add_argument("--pool", required=True, choices=["public", "private"])
    q.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    q.add_argument("--eval", default=os.path.join(HERE, "eval.json"))
    q.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", required=True)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    return {"select": cmd_select, "judge": cmd_judge, "apply": cmd_apply, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
