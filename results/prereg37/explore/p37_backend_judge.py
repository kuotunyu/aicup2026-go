"""prereg37 PLAN revision 1, section 2: the one E judgement of M37 + retuned prereg38 backend against the uploaded
v2-3 system (M36 + tune_D_B4.json). Seam: the v2-3 E mean must equal prereg38 eval.json bit for bit."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P38 = os.path.join(REPO, "results", "prereg38", "explore")
sys.path.insert(0, P38)
import p38_backend as B  # noqa: E402

pe, L, sim = B.pe, B.L, B.sim
P38_TUNE = os.path.join(P38, "tune_D_B4.json")
P38_EVAL = os.path.join(P38, "eval.json")
P36_TUNE = os.path.join(REPO, "results", "prereg36", "explore", "tune.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default=os.path.join(HERE, "tune_D_B4_M37.json"))
    ap.add_argument("--emb-E", default=os.path.join(REPO, "data", "prereg37", "emb", "step_160000", "store_E.npz"))
    ap.add_argument("--base-emb-E", default=os.path.join(REPO, "data", "prereg36", "emb", "step_40000", "store_E.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "eval_backend.json"))
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    if not L.git_committed(a.tune):
        raise SystemExit("commit the retuned backend before the judgement")
    tune = json.load(open(a.tune, encoding="utf-8"))
    base = json.load(open(P38_TUNE, encoding="utf-8"))
    rec38 = json.load(open(P38_EVAL, encoding="utf-8"))
    t36 = json.load(open(P36_TUNE, encoding="utf-8"))
    store = sim.Store(L.FEATS, load=())
    e_new = pe.Emb.load(a.emb_E, colour=store.color)
    e_base = pe.Emb.load(a.base_emb_E, colour=store.color)
    if e_new.sha != tune["emb"]["ckpt_sha256"]:
        raise SystemExit("--emb-E is not from the checkpoint the backend was retuned with")
    if e_base.sha != base["emb"]["ckpt_sha256"] or e_base.sha != t36["checkpoint"]["sha256"]:
        raise SystemExit("--base-emb-E is not the M36 checkpoint of v2-3")
    pools = pe.load_pools(pe.TERMS_DIR["E"], pe.E_SEEDS)
    num = B.judge_numbers(pools, e_new, tune["recommended"]["cfg"], e_base, base["recommended"]["cfg"])
    want = rec38["means"]["backend"]
    seam = {"recomputed": num["means"]["adopted_system"], "recorded_v2_3": want,
            "ok": num["means"]["adopted_system"] == want and num["per_pool"]["adopted_system"] == rec38["per_pool"]["backend"]}
    if not seam["ok"]:
        raise SystemExit("seam failed: %r" % seam)
    rec = {"format": "c10-prereg37 backend eval v1", "e_seeds": pe.E_SEEDS, "e_use": 3,
           "new": {"cfg": tune["recommended"]["cfg"], "ckpt_sha256": tune["emb"]["ckpt_sha256"]},
           "v2_3": {"cfg": base["recommended"]["cfg"], "ckpt_sha256": base["emb"]["ckpt_sha256"]},
           "tune_sha256": B.sha_json(a.tune), "git_head": L.git_head(), "seam_v2_3": seam}
    rec.update(num)
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("means", "backend_minus_adopted", "by_k", "adopt")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
