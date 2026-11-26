"""Review part C: is any inner game (question or profile game) also stored as a game of another player
(fit or a different inner player), which would put the question's own moves into a pi0 reference profile?
Exact move-sequence hash (all moves) and first-60-move hash. Training files only; guard installed."""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "u1"))
import u1_lib as U  # noqa
import hashlib, json
import numpy as np
DS = os.path.join(U.REPO, "data", "prereg04", "design_scratch")

def hashes(which, cap):
    z = np.load(os.path.join(DS, f"moves_{which}.npz"), allow_pickle=False)
    off, pid = z["off"], z["player_id"]
    z.close()
    mv = np.load(os.path.join(DS, f"moves_{which}_moves.npy"), mmap_mode="r")
    hs = []
    for i in range(len(off) - 1):
        s, e = int(off[i]), int(off[i + 1])
        if cap:
            e = min(e, s + cap)
        hs.append(hashlib.blake2b(np.asarray(mv[s:e]).tobytes(), digest_size=12).digest())
    return hs, pid, np.diff(off)

out = {}
for cap in (0, 60):
    hf, pf, lf = hashes("fit", cap)
    hi, pi, li = hashes("inner", cap)
    fset = {}
    for h, p in zip(hf, pf.tolist()):
        fset.setdefault(h, set()).add(p)
    in_fit = [h in fset for h in hi]
    iset = {}
    for h, p in zip(hi, pi.tolist()):
        iset.setdefault(h, set()).add(p)
    inner_other = [len(iset[h]) > 1 for h in hi]
    out[f"cap{cap}"] = {"inner_games": len(hi), "inner_game_seq_also_in_fit": int(sum(in_fit)),
                        "inner_game_seq_shared_by_2plus_inner_players": int(sum(inner_other)),
                        "fit_games": len(hf), "fit_distinct": len(fset), "inner_distinct": len(iset)}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "review_c.json"), "w"), indent=1)
