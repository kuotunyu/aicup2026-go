"""Write repro/ref_lda_L.npz: the L-player P1 LDA as fitted on the machine of the original run (Windows).

    python repro/make_tolerant_refs.py [--root REPO]

The registered competition-1 programs refit this LDA at every apply and require the SHA-256 of its arrays to be
the one in results/prereg43/explore/tune.json. On another platform the refit differs in the last bits and only a
hash is stored, so the size of the difference cannot be measured. This script stores the arrays themselves (joint /
B / W models: m, s, mu, sd, W), after checking that their SHA-256 is tune.json's; `--tolerant` compares against them
(repro/_tolerant.py). Calls the registered functions only (p39_eval.load_set / fit_models / models_sha).
data/local is never opened.
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _tolerant as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root")
    a = ap.parse_args()
    root = C.repo_root(a.root)
    os.chdir(root)
    sys.path.insert(0, os.path.join(root, "results", "prereg67", "explore"))
    import numpy as np
    import p67_eval as P67
    P67.BK.polite()
    if os.name != "nt":
        C.blas_one_thread_posix()
    want = P67.E45.v26()["lda_sha256"]
    models = P67.P39E.fit_models(P67.P39E.load_set("L"))
    got = P67.P39E.models_sha(models)
    if got != want:
        print(f"FAIL: the refit here has SHA-256 {got}, tune.json records {want}; nothing written")
        return 1
    np.savez(T.LDA_REF, **{f"{k}_{x}": np.ascontiguousarray(models[k][x], np.float64) for k, x in T.LDA_KEYS})
    again = P67.P39E.models_sha(T.load_lda_ref())
    print(f"{T.LDA_REF}: {os.path.getsize(T.LDA_REF):,} bytes, arrays SHA-256 {again} "
          f"({'= tune.json' if again == want else 'NOT tune.json'})")
    return 0 if again == want else 1


if __name__ == "__main__":
    sys.exit(main())
