"""Term registry for analysis (c) and fused score matrices (used by s4_terms.py, s5_combine.py and
s3_assign.py --term <spec json>)."""
import sys
sys.dont_write_bytecode = True
import json
import os

import numpy as np

import c1h_eval as E
import c1h_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))


def _G(name, G_cache):
    if name not in G_cache:
        G_cache[name] = L.load_family(name)
    return G_cache[name]


def _X(G_cache, name="style"):
    if name not in G_cache:
        G_cache[name] = np.load(os.path.join(L.SCRATCH, "dense_%s.npy" % name))
    return G_cache[name]


def term(name, rec, cache, G_cache):
    if name in ("base", "n20", "n30", "deep", "prefix", "opp", "joseki", "joseki_own", "resp"):
        return E.cos_term(_G(name, G_cache), rec)
    if name in ("style", "style2"):
        return E.style_ll(_X(G_cache, name), rec)
    if name.startswith("gw"):                      # gw<beta>: per-game log-softmax, base tokens
        return E.gamewise_logp(_G("base", G_cache), rec, float(name[2:]))
    if name.startswith("gwn20_"):
        return E.gamewise_logp(_G("n20", G_cache), rec, float(name[6:]))
    if name.startswith("knn"):                     # knn<top>: game-to-game, base tokens
        return E.knn_term(_G("base", G_cache), rec, top=int(name[3:]))
    if name.startswith("pknn"):                    # pknn<top>: game-to-game, prefix tokens
        return E.knn_term(_G("prefix", G_cache), rec, top=int(name[4:]))
    if name.startswith("lda"):                     # lda<d>: c1h_lda (added in the continuation run)
        import c1h_lda as LD
        return LD.lda_term(rec, cache, int(name[3:]))
    raise KeyError(name)


TERMS = ["base", "n20", "n30", "deep", "prefix", "opp", "joseki", "joseki_own", "resp", "style",
         "gw10", "knn1", "knn3", "pknn1"]


def fused_matrix(spec, rec, cache, G_cache):
    """spec: path or name of a json {"terms": [...], "weights": [...], "post": null|"csls10"|"colc"}."""
    if not os.path.exists(spec):
        spec = os.path.join(HERE, spec if spec.endswith(".json") else spec + ".json")
    with open(spec, encoding="utf-8") as f:
        d = json.load(f)
    S = E.fuse([term(t, rec, cache, G_cache) for t in d["terms"]], d["weights"])
    post = d.get("post")
    if post and post.startswith("csls"):
        S = E.csls(S, int(post[4:]))
    elif post == "colc":
        S = E.col_center(S)
    return S
