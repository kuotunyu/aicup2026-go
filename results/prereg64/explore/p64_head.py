"""prereg64: c49 head rows (frozen.json classes.head) from the checkpoint, CPU only. Run with .venv-gpu (torch).

    GPUPY results/prereg64/explore/p64_head.py   -> data/prereg64/c49_head.npz, results/prereg64/explore/head.json

The rows must equal Gemini's data/advice/gemini_1121/c49_head.npz bit for bit (weight, class ids, class ranks).
"""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg36", "explore"),
          os.path.join(REPO, "results", "prereg49", "explore"), os.path.join(REPO, "results", "prereg58", "explore")):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import p49_lib as L49  # noqa: E402
import p58_lib as K58  # noqa: E402

CKPT = os.path.join(REPO, "data", "prereg49", "S0", "ckpt", "step_320000.pt")
OUT = os.path.join(REPO, "data", "prereg64", "c49_head.npz")
REC = os.path.join(HERE, "head.json")
GEMINI = os.path.join(REPO, "data", "advice", "gemini_1121", "c49_head.npz")


def main():
    torch.set_num_threads(2)
    if os.path.exists(OUT) or os.path.exists(REC):
        raise SystemExit("head exists")
    cache = L49.Cache49()
    T = L49.t49(cache, L49.e_players())
    rank_of = {}
    for p, r in zip(np.asarray(cache.pid).astype(str).tolist(), np.asarray(cache.rank).astype(str).tolist()):
        if rank_of.setdefault(p, r) != r:
            raise SystemExit("player with two ranks")
    cr = K58.rank_index([rank_of[p] for p in T])
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    W = ck["head"]["weight"].detach().cpu().float().numpy()
    if len(W) != len(T):
        raise SystemExit("head rows %d != T49 %d" % (len(W), len(T)))
    with np.load(K58.guard(GEMINI), allow_pickle=False) as z:
        g = {k: np.asarray(z[k]) for k in z.files}
    same = {"weight": bool(np.array_equal(g["weight"], W)),
            "class_id": bool(np.array_equal(g["class_id"].astype(str), np.array(T))),
            "class_rank": bool(np.array_equal(g["class_rank"].astype(np.int64), cr))}
    if not all(same.values()):
        raise SystemExit("differs from Gemini's c49_head.npz: %s" % same)
    sha_ck = K58.sha256_file(CKPT)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT + ".tmp.npz", weight=W.astype(np.float32), class_id=np.array(T), class_rank=cr,
             ckpt_sha256=np.array(sha_ck), ckpt_step=np.array(int(ck["step"])))
    os.replace(OUT + ".tmp.npz", OUT)
    rec = {"format": "c10-prereg64 head v1", "checkpoint": "data/prereg49/S0/ckpt/step_320000.pt",
           "ckpt_sha256": sha_ck, "ckpt_step": int(ck["step"]), "classes": int(len(W)), "dim": int(W.shape[1]),
           "classes_per_rank": np.bincount(cr, minlength=10).tolist(), "equal_to_gemini_c49_head": same,
           "out": "data/prereg64/c49_head.npz", "out_sha256": K58.sha256_file(OUT),
           "class_ids_sha256": hashlib.sha256("\n".join(T).encode()).hexdigest()}
    K58.write_json(REC, rec)
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
