"""--tolerant mode of the repro/ entry points (C10, AI CUP 2026 Go). Nothing here computes a prediction.

The frozen programs stop when a recomputed intermediate result is not byte for byte the stored one. The stored
results were computed on Windows; on Linux numpy / scipy come with another BLAS / libm build and the last bits of
some floating-point results differ, so the run stops at a seam before any final CSV exists.

With --tolerant, and only then, the entry point replaces - inside its own process, no file is edited - the exact
comparisons listed below by comparisons with a tolerance, and records for each one what the difference really was:

    numeric seam    passes when every value satisfies |got - ref| <= atol + rtol * |ref| (ATOL / RTOL below; a seam
                    that defines its own tolerance keeps it). Beyond the tolerance the run still stops.
    discrete seam   (a prediction file compared byte for byte) the differing rows are counted and recorded and the
                    run goes on, so that the final CSV exists and can be compared row by row. This never turns a
                    difference into a PASS: PASS is still only "the final CSV has the SHA-256 of the uploaded file".

Everything is written to <out>/tolerant_report.json (also when the run stops).
"""
import csv
import io
import json
import os

import numpy as np

ATOL = 1e-9
RTOL = 1e-9
HERE = os.path.dirname(os.path.abspath(__file__))
LDA_REF = os.path.join(HERE, "ref_lda_L.npz")
LDA_KEYS = [(k, a) for k in ("joint", "B", "W") for a in ("m", "s", "mu", "sd", "W")]


class Report:
    def __init__(self, chain, out):
        self.chain, self.path, self.context, self.seams = chain, os.path.join(out, "tolerant_report.json"), "", []

    def add(self, seam, kind, strict_equal, within, **more):
        e = {"seam": seam, "context": self.context, "kind": kind, "strict_equal": bool(strict_equal),
             "within_tolerance": bool(within), "relaxed": bool(within and not strict_equal)}
        e.update(more)
        self.seams.append(e)
        tag = "exact" if strict_equal else ("RELAXED" if within else "BEYOND TOLERANCE")
        nums = ", ".join(f"{k} {e[k]:.3g}" if isinstance(e[k], float) else f"{k} {e[k]}"
                         for k in ("max_abs_diff", "max_rel_diff", "rows_different") if e.get(k) is not None)
        print(f"[tolerant] {seam}{' (' + self.context + ')' if self.context else ''}: {tag}{'; ' + nums if nums else ''}",
              flush=True)
        self.write()
        return e

    def write(self, final=None):
        num = [e for e in self.seams if e["kind"] != "discrete"]
        dis = [e for e in self.seams if e["kind"] == "discrete"]
        rec = {"format": "c10-repro tolerant report v1", "chain": self.chain,
               "rule": "numeric: |got - ref| <= atol + rtol * |ref| (a seam's own tolerance is kept); discrete: "
                       "differing rows are recorded and the run goes on; PASS is only the final CSV's SHA-256",
               "default_atol": ATOL, "default_rtol": RTOL,
               "seams_checked": len(self.seams), "seams_relaxed": sum(e["relaxed"] for e in self.seams),
               "numeric_all_within_tolerance": all(e["within_tolerance"] for e in num),
               "numeric_max_abs_diff": max([e.get("max_abs_diff") or 0.0 for e in num], default=0.0),
               "discrete_rows_different": sum(e.get("rows_different") or 0 for e in dis),
               "seams": self.seams}
        if final is not None:
            rec["final_csv"] = final
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            json.dump(rec, f, indent=1)


# ---------------------------------------------------------------- comparisons
def _rows(data):
    return list(csv.reader(io.StringIO(data.replace(b"\r\n", b"\n").decode("utf-8"))))


def csv_rows_diff(a, b, first=10):
    """Rows that differ between two CSV byte strings (CRLF folded) -> (count, rows a, rows b, first differences)."""
    ra, rb = _rows(a), _rows(b)
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(ra, rb)) if x != y]
    return len(diff) + abs(len(ra) - len(rb)), len(ra), len(rb), diff[:first]


def csv_numeric_diff(a, b, atol=ATOL, rtol=RTOL):
    """Cell by cell: equal text, or both numbers within the tolerance. -> dict(max_abs_diff, max_rel_diff,
    cells_different, structure_problems, within)."""
    ra, rb = _rows(a), _rows(b)
    max_abs = max_rel = 0.0
    cells = bad = 0
    struct = abs(len(ra) - len(rb))
    for x, y in zip(ra, rb):
        if len(x) != len(y):
            struct += 1
            continue
        for u, v in zip(x, y):
            if u == v:
                continue
            try:
                fu, fv = float(u), float(v)
            except ValueError:
                struct += 1
                continue
            cells += 1
            d = abs(fu - fv)
            if d != d:                                        # NaN on one side
                struct += 1
                continue
            max_abs = max(max_abs, d)
            if fv != 0.0:
                max_rel = max(max_rel, d / abs(fv))
            if d > atol + rtol * abs(fv):
                bad += 1
    return {"max_abs_diff": max_abs, "max_rel_diff": max_rel, "cells_different": cells, "cells_beyond_tolerance": bad,
            "structure_problems": struct, "within": struct == 0 and bad == 0}


def array_diff(got, ref, atol=ATOL, rtol=RTOL):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    if got.shape != ref.shape:
        return {"shape": [list(got.shape), list(ref.shape)], "within": False}
    d = np.abs(got - ref)
    nz = np.abs(ref) > 0
    return {"max_abs_diff": float(d.max()) if d.size else 0.0,
            "max_rel_diff": float((d[nz] / np.abs(ref[nz])).max()) if nz.any() else 0.0,
            "within": bool(np.isfinite(got).all() and (d <= atol + rtol * np.abs(ref)).all())}


class _SeamBytes(bytes):
    """bytes whose == / != go through a seam comparison (set per instance); everything else is plain bytes."""
    _cmp = None

    def __eq__(self, other):
        if self._cmp is None or not isinstance(other, bytes):
            return bytes.__eq__(self, other)
        return self._cmp(bytes(self), bytes(other), getattr(other, "_name", None))

    def __ne__(self, other):
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    __hash__ = bytes.__hash__


def _seam_bytes(data, cmp, name=None):
    b = _SeamBytes(data)
    b._cmp, b._name = cmp, name
    return b


# ---------------------------------------------------------------- competition 2 (tools/t69_rank.py)
def install_comp2(t69, rep):
    """seam A (and every other t25.rows_compare): the tool's own tolerance (1e-12, rule A / rule B levels equal)
    instead of byte equality as well. seam T: probabilities within ATOL / RTOL; the merge-off submission is a
    discrete seam."""
    t25 = t69.t25
    orig_rows_compare = t25.rows_compare

    def rows_compare(rows, ref_path):
        rec = orig_rows_compare(rows, ref_path)
        rep.add("t25.rows_compare vs " + os.path.relpath(ref_path, t69.REPO).replace(os.sep, "/"), "numeric",
                rec["byte_equal"], rec["equal"], max_abs_diff=float(rec["max_abs_diff"]), tol=rec["tol"],
                tolerance_source="the tool's own (PREREG_25 4.2)", rule_a_differences=rec["rule_a_differences"],
                rule_b_differences=rec["rule_b_differences"], problems=rec["problems"])
        if rec["equal"] and not rec["byte_equal"]:
            rec["byte_equal_strict"], rec["byte_equal"], rec["tolerant"] = False, True, True
        return rec

    t25.rows_compare = rows_compare
    orig_bytes_of = t69.bytes_of

    def cmp_seam_t(a, b, other_name):
        name = other_name or ""
        if os.path.basename(name) == "rank_test_probs.csv":
            d = csv_numeric_diff(a, b)
            rep.add("seam T: Q64 probabilities (merge off) vs results/test64/rank/rank_test_probs.csv", "numeric",
                    a == b, d["within"], atol=ATOL, rtol=RTOL, **{k: v for k, v in d.items() if k != "within"})
            return d["within"]
        n, la, lb, first = csv_rows_diff(a, b)
        rep.add("seam T: rule B predictions (merge off) vs results/test64/rank/submission_rank.csv", "discrete",
                a == b, n == 0, rows_different=n, rows=[la, lb], first_differences=first)
        return True                                           # recorded; the final CSV is what is judged

    def bytes_of(path):
        data = orig_bytes_of(path)
        if "tmp_seam_t" in os.path.normpath(path).split(os.sep) or os.path.dirname(os.path.abspath(path)) == \
                os.path.abspath(t69.RANK64):
            return _seam_bytes(data, cmp_seam_t, path)
        return data

    t69.bytes_of = bytes_of


# ---------------------------------------------------------------- competition 1 (results/prereg67/explore/p67_eval.py)
def load_lda_ref():
    with np.load(LDA_REF, allow_pickle=False) as z:
        return {k: {a: np.asarray(z[f"{k}_{a}"], np.float64) for a in ("m", "s", "mu", "sd", "W")}
                for k in ("joint", "B", "W")}


def lda_diff(models, ref, atol=ATOL, rtol=RTOL):
    """The refitted L-LDA against the stored Windows fit, array by array. An eigenvector is defined up to its sign,
    so W is also compared after giving each column the sign of the reference column (recorded separately; the
    tolerance is applied to W as it is)."""
    per, within, max_abs, max_rel = {}, True, 0.0, 0.0
    for k, a in LDA_KEYS:
        d = array_diff(models[k][a], ref[k][a], atol, rtol)
        if a == "W" and "shape" not in d:
            g, r = np.asarray(models[k][a], np.float64), ref[k][a]
            sgn = np.where((g * r).sum(0) < 0, -1.0, 1.0)
            d["columns_with_flipped_sign"] = int((sgn < 0).sum())
            d["max_abs_diff_sign_aligned"] = float(np.abs(g * sgn - r).max())
        per[f"{k}.{a}"] = d
        within &= d["within"]
        max_abs, max_rel = max(max_abs, d.get("max_abs_diff", np.inf)), max(max_rel, d.get("max_rel_diff", np.inf))
    return {"max_abs_diff": float(max_abs), "max_rel_diff": float(max_rel), "arrays": per, "within": bool(within)}


def install_comp1(P67, rep):
    """The L-LDA refit (its SHA-256 must be prereg43 tune.json's): compared with the stored Windows fit
    repro/ref_lda_L.npz within ATOL / RTOL. The rebuilt v2-8 top5 (byte for byte results/prereg66/apply/<pool>/
    top5.csv): a discrete seam."""
    P39E, P63 = P67.P39E, P67.P63
    orig_sha = P39E.models_sha
    ref = load_lda_ref()
    ref_sha = orig_sha(ref)

    def models_sha(models):
        real = orig_sha(models)
        if real == ref_sha:
            rep.add("L-LDA refit vs prereg43 tune.json (SHA-256 of the arrays)", "numeric", True, True,
                    max_abs_diff=0.0, sha256=real)
            return real
        d = lda_diff(models, ref)
        rep.add("L-LDA refit vs prereg43 tune.json (SHA-256 of the arrays)", "numeric", False, d["within"],
                atol=ATOL, rtol=RTOL, sha256_refit=real, sha256_reference=ref_sha,
                reference="repro/ref_lda_L.npz (the Windows fit; its SHA-256 is the one tune.json records)",
                **{k: v for k, v in d.items() if k != "within"})
        return ref_sha if d["within"] else real

    P39E.models_sha = models_sha
    orig_lf = P63.lf

    def cmp_top5(a, b, _other):
        n, la, lb, first = csv_rows_diff(a, b)
        rep.add("v2-8 top5 rebuilt vs results/prereg66/apply/<pool>/top5.csv", "discrete", a == b, n == 0,
                rows_different=n, rows=[la, lb], first_differences=first)
        return True                                           # recorded; the final CSV is what is judged

    def lf(b):
        out = orig_lf(b)
        return _seam_bytes(out, cmp_top5) if out.startswith(b"question_id,top_1") else out

    P63.lf = lf
    return ref_sha
