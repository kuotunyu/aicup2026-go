"""prereg09 mapping exploration: the inner evaluation of a candidate P1TO mapping, through tools/p08_rank_fuse.py's
own tune machinery (preview only; nothing is written into results/prereg08).

Inner(): loads u1_lib's inner inputs (training players only; u1_lib / hs_lib install the data/local guard), the HS
baseline (p1_fuse.hs_baseline, as tune), HS usable mask (pilot chunks), then
    Q02H  = run_terms(Q02H_TERMS, HS)                       seam: 0.6242752597994481 (P1REF)
    Q08   = run_terms(Q08_TERMS, HS, 4 x lp of the frozen p1_map.npz)   tune: 0.6986195853659671
Inner.evaluate(lp_rows, scale) -> the same fusion (5-fold OOF calibration, p04 gen path, weights refit per fold)
with the candidate's lp in place of lp_P1TO; scores on inner eval (rank-balanced) and on all 120,544 single-colour
questions; paired PlayerBootstrap differences vs Q08 and vs Q02H (rf.N_BOOT draws, same seeds as tune).
"""

import json
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
TMP = os.path.join(REPO, "data", "prereg09", "tmp", "mapping")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
import p08_rank_fuse as R  # noqa: E402  (pin checks of the p07 tools)

rf, p3, p4, P7 = R.rf, R.p3, R.p4, R.P7
Q08_TUNE = 0.6986195853659671
Q08_TUNE_MONO = 0.6946737091777063


def ci(d):
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


class Inner:
    def __init__(self, verbose=True):
        t0 = time.time()
        self.PF = R.import_pilot_fuse()
        _L1, self.F = R.import_p1()
        self.U = R._import_u1()
        d = self.U.load_inputs(verbose=False)
        self.d = d
        inner = d["inner"]
        self.n_inner = inner["keys"]["n"]
        lp_hs, _cov, _rec = self.F.hs_baseline(self.PF, self.U, d)
        d["lp"]["HS"] = lp_hs
        hs_usable, _f, _c = R.pilot_hs_usable(self.PF, self.n_inner)
        self.E_hs, _ = R.masked_inner(d, lp_hs, hs_usable)
        meta = np.load(os.path.join(TMP, "inner_meta.npz"))
        self.ref = meta["ref"]
        self.covered = np.zeros(self.n_inner, dtype=bool)
        self.covered[self.ref] = True
        self.Xe, self.Xm, self.ms = inner["X_eval"], inner["X_mono"], inner["ms"]
        self.boot, self.mboot = inner["boot"], inner["mono_boot"]
        self.h = self._run(R.Q02H_TERMS, {"HS": self.E_hs})
        if abs(self.h["rb"] - R.P1REF["Q02H"]) > R.TOL:
            raise RuntimeError(f"Q02H seam {self.h['rb']!r}")
        # Q08 with the frozen map (float32 stored standardised features -> tiny lp differences vs tune)
        with np.load(os.path.join(REPO, "results", "prereg08", "rank", "p1_map.npz")) as z:
            self.W, self.b = z["W"], z["b"]
        lp_rows = self.lr_rows(self.W, self.b)
        self.q08 = self.evaluate_raw(lp_rows, 4.0)
        if verbose:
            print(f"Inner ready in {time.time() - t0:.0f}s: Q02H {self.h['rb']:.6f}, Q08 {self.q08['rb']!r} "
                  f"(tune {Q08_TUNE!r}), mono {self.q08['mono_rb']!r} (tune {Q08_TUNE_MONO!r})", flush=True)

    def lr_rows(self, W, b, block=20000):
        X = np.load(os.path.join(TMP, "inner_X.npy"), mmap_mode="r")
        out = np.zeros((X.shape[0], 10))
        for s in range(0, X.shape[0], block):
            z = np.asarray(X[s:s + block], dtype=np.float64) @ W + b
            out[s:s + block] = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
        return out

    def _run(self, terms, Emap):
        P, Pm, folds, _aux = R.run_terms(self.d, terms, Emap)
        m, sc = rf.metrics(rf.rule_b_np(P), self.Xe.y, self.Xe.n, self.Xe.player)
        mrec, msc = p3.mono_score_record(rf.rule_b_np(Pm), self.Xm, self.ms, self.mboot)
        w = [p4.gen_params(fr["theta"], terms)["w"] for fr in folds]
        return {"rb": m["rank_balanced"], "qm": m["question_mean"], "per_level": m["per_level"], "sc": sc,
                "mono_rb": mrec["rank_balanced"], "msc": msc,
                "mono_by_colour": {c: mrec["by_colour"][c]["rank_balanced"] for c in mrec["by_colour"]},
                "weights": w}

    def evaluate_raw(self, lp_rows, scale):
        lp = np.zeros((self.n_inner, 10))
        lp[self.ref] = lp_rows
        E, _ = R.masked_inner(self.d, lp, self.covered, scale=scale)
        return self._run(R.Q08_TERMS, {"HS": self.E_hs, "P1TO": E})

    def evaluate_multi_raw(self, lp_rows_list, scale):
        """several P1 terms at once (each its own fused weight): Q02H_TERMS + [P1TO, P1X1, ...]."""
        Emap, terms = {"HS": self.E_hs}, list(R.Q02H_TERMS)
        for i, lr_ in enumerate(lp_rows_list):
            lp = np.zeros((self.n_inner, 10))
            lp[self.ref] = lr_
            t = "P1TO" if i == 0 else f"P1X{i}"
            Emap[t], _ = R.masked_inner(self.d, lp, self.covered, scale=scale)
            terms.append(t)
        return self._run(terms, Emap)

    def compare(self, r, base):
        dd = self.boot.stats(r["sc"]) - self.boot.stats(base["sc"])
        md = p3.mono_diff_record(self.ms, self.mboot, r["msc"], base["msc"])
        return {"inner_diff": r["rb"] - base["rb"], "inner_ci95": ci(dd),
                "mono_diff": md["rank_balanced"], "mono_ci95": md["ci95"],
                "mono_by_colour": {c: (md["by_colour"][c] or {}).get("rank_balanced") for c in md["by_colour"]},
                "mono_by_colour_ci95": {c: (md["by_colour"][c] or {}).get("ci95") for c in md["by_colour"]}}

    def evaluate(self, lp_rows, scale=4.0, name=""):
        t = time.time()
        r = (self.evaluate_multi_raw(lp_rows, scale) if isinstance(lp_rows, (list, tuple))
             else self.evaluate_raw(lp_rows, scale))
        out = {"scale": scale, "inner_rank_balanced": r["rb"], "inner_question_mean": r["qm"],
               "mono_rank_balanced": r["mono_rb"], "mono_by_colour": r["mono_by_colour"],
               "fusion_weights_by_fold": r["weights"], "per_level": r["per_level"],
               "vs_Q08": self.compare(r, self.q08), "vs_Q02H": self.compare(r, self.h),
               "seconds": round(time.time() - t, 1)}
        v = out["vs_Q08"]
        print(f"[{name} x{scale:g}] inner {r['rb']:.4f} ({v['inner_diff']:+.4f} {np.round(v['inner_ci95'], 4)}) "
              f"mono {r['mono_rb']:.4f} ({v['mono_diff']:+.4f} {np.round(v['mono_ci95'], 4)} "
              f"B {v['mono_by_colour'].get('B', 0):+.4f} W {v['mono_by_colour'].get('W', 0):+.4f}) "
              f"{out['seconds']:.0f}s", flush=True)
        return out

    def baseline_record(self):
        return {"Q02H": {"inner_rank_balanced": self.h["rb"], "mono_rank_balanced": self.h["mono_rb"]},
                "Q08_recomputed": {"inner_rank_balanced": self.q08["rb"], "mono_rank_balanced": self.q08["mono_rb"],
                                   "mono_by_colour": self.q08["mono_by_colour"],
                                   "fusion_weights_by_fold": self.q08["weights"],
                                   "tune_inner": Q08_TUNE, "tune_mono": Q08_TUNE_MONO,
                                   "abs_diff_vs_tune": abs(self.q08["rb"] - Q08_TUNE)},
                "Q08_vs_Q02H": self.compare(self.q08, self.h)}


def dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    os.replace(tmp, path)
