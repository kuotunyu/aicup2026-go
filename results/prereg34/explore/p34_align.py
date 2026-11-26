"""prereg34 exploration (competition 2), stage 1b: OGS rank -> official level alignment (PLAN.md 3, steps 1-4).
CPU only (.venv; numpy forward of the frozen S25, no torch). Only OGS data: no inner data, no validation label, never the
answers folder, never tools/score.py.

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg34/explore
    $S/p34_align.py align           -> $S/align_record.json (once; code committed and clean; reads data/prereg34/prep)
    $S/p34_align.py align --smoke   -> data/prereg34/smoke/align_record.json (synthetic OGS prep of p34_prep.py synth:
                                       pipeline only, never a result)

Rule (PLAN.md 3, fixed here where PLAN leaves a detail open; every choice is also written into the record):
1. S25 = the frozen results/prereg25/rank/s25_seed{0,1,2}.npz (SHA = prereg25 train_record.json; p29_lib.s25_models).
   Features = the OGS rows standardised with the frozen Q09 map's mu_x / sd_x (= S25's standardisation).
2. One question per (player, colour): that colour's usable rows, at most QUESTION_MAX = 20; when a colour has more, the
   NEWEST 20 by start_time (ties: the larger numeric OGS game id is the newer). Question log-probabilities = the S25
   set term's lp: lse mean over the 3 seeds of head(mean of h over the question's games) (p19_lib.set_term, unchanged).
3. Player log-probabilities = the arithmetic mean of the two colour questions' log-probability vectors (only one
   colour when the other has no usable row); renormalised (softmax of the mean) so that the probabilities sum to 1
   (the mean of two normalised log-probability vectors is not normalised); e = sum_k k * p_k with D = 0 .. 6D = 9.
4. f = isotonic (non-decreasing) least-squares fit of e on the player's OGS median (the stage 1a window median), equal
   weights per player, pool-adjacent-violators (no sklearn in .venv); tied medians are pooled first. f is evaluated
   at each player's own median.
5. label = floor(f + 0.5) (round half up); a player with f < -0.5 or f >= 9.5 is dropped (half-open so that every kept
   label is in 0 .. 9; f lies in [0, 9] by construction, so nobody is dropped in practice).
Record only: per OGS bin (floor of the median, 15 kyu .. 7 dan, p34_ogs.bin_of) the players, mean e and mean f; the f
blocks (knots); players per official level. The per-player labels are written into the record (p34_train reads them
from the committed record).
"""

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P19 = os.path.join(REPO, "results", "prereg19", "explore")
P25 = os.path.join(REPO, "results", "prereg25", "explore")
P29 = os.path.join(REPO, "results", "prereg29", "explore")
P23 = os.path.join(REPO, "results", "prereg23", "explore")
for p in (os.path.join(REPO, "tools"), P19, P23, P25, P29, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p34_prep as PR  # noqa: E402
import p34_ogs as OGS  # noqa: E402  (bin_of: the stage 1a bins)
import p19_lib as K19  # noqa: E402
import p29_lib as L29  # noqa: E402
import p09_forward as FW  # noqa: E402

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")   # = tools/c10data.LEVELS (index = level)
QUESTION_MAX = 20
LABEL_LO, LABEL_HI = -0.5, 9.5
ALIGN_FORMAT = "c10-prereg34 explore align v1"
RECORD = os.path.join(HERE, "align_record.json")
SMOKE_RECORD = os.path.join(PR.SMOKE_DIR, "align_record.json")
Q09_MAP = PR.Q09_MAP
Q09_FROZEN = os.path.join(REPO, "results", "prereg09", "rank", "frozen_config.json")
OWN_FILES = ("PLAN.md", "p34_prep.py", "p34_align.py", "p34_ogs.py")
IMPORTED = ("results/prereg19/explore/p19_lib.py", "results/prereg25/explore/p25_lib.py",
            "results/prereg29/explore/p29_lib.py", "tools/p09_forward.py", "tools/p09_map.py",
            "results/prereg09/rank/p1e_map.npz", "results/prereg09/rank/frozen_config.json",
            "results/prereg25/explore/train_record.json", "results/prereg25/rank/s25_seed0.npz",
            "results/prereg25/rank/s25_seed1.npz", "results/prereg25/rank/s25_seed2.npz")
FORBIDDEN_SUBSTR = "data/lo" + "cal"


def log(*x):
    print(*x, flush=True)


rel = PR.rel


def git(*args):
    import subprocess
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def sha_lf(path):
    import hashlib
    with open(PR.guard(path), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def code_state(own, imported, here=HERE):
    """Head, dirty files and LF SHA of the own files, the listed imports and every loaded repository module."""
    files = [rel(os.path.join(here, f)) for f in own] + list(imported)
    files = sorted(set(files) | L29.loaded_repo_modules())
    if any(FORBIDDEN_SUBSTR in f for f in files):
        raise SystemExit("refused path in code_state")
    missing = [x for x in files if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    return {"head": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain", "--", *files),
            "sha256_lf": {x: sha_lf(os.path.join(REPO, *x.split("/"))) for x in files}}


def committed_clean(path):
    r_ = rel(path)
    return bool(os.path.exists(path) and git("ls-files", "--", r_) == r_ and not git("status", "--porcelain", "--", r_))


# ---------------------------------------------------------------- pure parts (unit-tested)
def game_number(gid):
    s = str(gid)
    if not s.startswith(PR.OGS_PREFIX) or not s[len(PR.OGS_PREFIX):].isdigit():
        raise ValueError(f"not an OGS game id: {gid}")
    return int(s[len(PR.OGS_PREFIX):])


def questions(meta, qmax=QUESTION_MAX):
    """-> (players sorted, {player: {colour: row indices}}): per colour the newest qmax rows by (start_time, game
    number), returned in ascending row order (a set mean does not depend on the order)."""
    pl = np.asarray(meta["players"]).astype(str)
    col = np.asarray(meta["color"]).astype(str)
    t = np.asarray(meta["start_time"]).astype(np.int64)
    gn = np.array([game_number(g) for g in np.asarray(meta["game_id"]).astype(str)], dtype=np.int64)
    out = {}
    for i in range(len(pl)):
        out.setdefault(pl[i], {}).setdefault(col[i], []).append(i)
    for p, d in out.items():
        for c in list(d):
            rows = np.array(d[c], dtype=np.int64)
            order = np.lexsort((gn[rows], t[rows]))[::-1]       # newest first
            d[c] = np.sort(rows[order[:qmax]])
    return sorted(out), out


def player_e(lp_by_colour):
    """lp_by_colour: list of 1 or 2 log-probability vectors [10] -> (renormalised p [10], e)."""
    m = np.mean(np.asarray(lp_by_colour, dtype=np.float64), axis=0)
    m = m - m.max()
    p = np.exp(m)
    p = p / p.sum()
    return p, float(np.dot(np.arange(len(p)), p))


def pav(x, y):
    """Isotonic (non-decreasing) least squares of y on x with equal weights (ties of x pooled first).
    -> (unique x ascending, fitted value per unique x, fitted value per input point)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ux, inv = np.unique(x, return_inverse=True)
    s = np.bincount(inv, weights=y, minlength=len(ux))
    w = np.bincount(inv, minlength=len(ux)).astype(np.float64)
    bs, bw, bn = [], [], []                    # block sums, weights, number of unique x
    for i in range(len(ux)):
        bs.append(s[i])
        bw.append(w[i])
        bn.append(1)
        while len(bs) > 1 and bs[-2] / bw[-2] > bs[-1] / bw[-1]:
            bs[-2] += bs[-1]
            bw[-2] += bw[-1]
            bn[-2] += bn[-1]
            del bs[-1], bw[-1], bn[-1]
    fu = np.concatenate([np.full(n, a / b) for a, b, n in zip(bs, bw, bn)]) if bs else np.zeros(0)
    return ux, fu, fu[inv]


def labels_of(f):
    """-> (label int [N] (-1 where dropped), keep bool [N]): label = floor(f + 0.5), kept iff -0.5 <= f < 9.5."""
    f = np.asarray(f, dtype=np.float64)
    keep = (f >= LABEL_LO) & (f < LABEL_HI)
    lab = np.where(keep, np.floor(f + 0.5), -1).astype(np.int64)
    return lab, keep


def knots(ux, fu):
    """Constant pieces of the fitted step function: [[x_first, x_last, f], ...]."""
    out = []
    for x, v in zip(ux.tolist(), fu.tolist()):
        if out and out[-1][2] == v:
            out[-1][1] = x
        else:
            out.append([x, x, v])
    return out


def align(pars, H_list, meta):
    """pars / H_list: the S25 seeds and their per-row h on the OGS rows -> per-player arrays and the f fit."""
    players, q = questions(meta)
    groups, owner = [], []
    for i, p in enumerate(players):
        for c in K19.COLOURS:
            if c in q[p]:
                groups.append([int(x) for x in q[p][c]])
                owner.append(i)
    n = len(meta["players"])
    _E, _per, lp, has = K19.set_term(pars, H_list, groups, n, scale=1.0)
    if not has.all():
        raise SystemExit("an empty question")
    by_p = [[] for _ in players]
    for g, i in enumerate(owner):
        by_p[i].append(lp[g])
    med = {}
    for p, m in zip(np.asarray(meta["players"]).astype(str), np.asarray(meta["player_median"], dtype=np.float64)):
        med.setdefault(p, float(m))
    x = np.array([med[p] for p in players])
    e = np.array([player_e(v)[1] for v in by_p])
    ux, fu, f = pav(x, e)
    lab, keep = labels_of(f)
    nq = {c: int(sum(c in q[p] for p in players)) for c in K19.COLOURS}
    qsize = [len(g) for g in groups]
    return {"players": players, "median": x, "e": e, "f": f, "label": lab, "keep": keep, "ux": ux, "fu": fu,
            "questions_per_colour": nq, "question_rows": {"min": int(min(qsize)), "max": int(max(qsize)),
                                                          "total": int(sum(qsize))}}


def per_bin(res):
    out = {}
    bins = np.array([OGS.bin_of(m) for m in res["median"]], dtype=np.int64)
    for b in range(OGS.BIN_LO, OGS.BIN_HI + 1):
        m = bins == b
        name = "%dk" % (30 - b) if b < 30 else "%dd" % (b - 29)
        out[name] = {"bin": b, "players": int(m.sum()),
                     "e_mean": float(res["e"][m].mean()) if m.any() else None,
                     "f_mean": float(res["f"][m].mean()) if m.any() else None,
                     "labels": np.bincount(res["label"][m & res["keep"]], minlength=10).tolist()}
    return out


# ---------------------------------------------------------------- command
def cmd_align(a):
    t0 = time.time()
    smoke = bool(a.smoke)
    out = SMOKE_RECORD if smoke else RECORD
    prep_dir = PR.SMOKE_PREP if smoke else PR.PREP
    if os.path.exists(out):
        raise SystemExit(f"{rel(out)} exists: the alignment is done once")
    import p23_eval as E23  # noqa: E402  (load_npz_model; lazy so that p34_train does not need it)
    cs = code_state(OWN_FILES, IMPORTED + ("results/prereg23/explore/p23_eval.py",))
    if not smoke and cs["dirty"]:
        raise SystemExit(f"the prereg34 files or the imported tools are not committed and clean: {cs['dirty']}")
    rows = PR.OgsRows(prep_dir)
    fsha = rows.files_sha256()
    if fsha != rows.record["files_sha256"]:
        raise SystemExit("ogs_X.npy / ogs_meta.npz differ from prep_record.json")
    if bool(rows.record.get("smoke")) != smoke:
        raise SystemExit("prep smoke flag differs from this run")
    mp = FW.load_map(Q09_MAP)
    map_sha = PR.sha_file(Q09_MAP)
    with open(Q09_FROZEN, encoding="utf-8") as f:
        if map_sha != json.load(f)["map"]["sha256"]:
            raise SystemExit("p1e_map.npz differs from frozen_config.json")
    pars, seam_f = L29.s25_models(K19, E23.load_npz_model)
    if not seam_f["ok"]:
        raise SystemExit(f"seam F (frozen S25) failed: {seam_f}")
    # per-row h of every seed (float64 forward on the standardised rows; = p23_eval.s19_term's per-game h)
    H = [np.zeros((rows.n, K19.H_DIM)) for _ in pars]
    for s in range(0, rows.n, PR.BLOCK):
        e = min(s + PR.BLOCK, rows.n)
        X = FW.standardise_x(np.asarray(rows.X[s:e], dtype=np.float64), mp["mu_x"], mp["sd_x"])
        for par, Hs in zip(pars, H):
            Hs[s:e] = K19.h_np(par, X)
    res = align(pars, H, rows.meta)
    lab = res["label"]
    keep = res["keep"]
    labels = {p: int(l_) for p, l_, k in zip(res["players"], lab.tolist(), keep.tolist()) if k}
    rec = {"format": ALIGN_FORMAT, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "smoke": smoke,
           "smoke_note": "synthetic OGS players with noise features: never a result" if smoke else None,
           "code": cs,
           "what": "PLAN.md 3 steps 1-4: frozen S25 (3 seeds, lse mean) expected level e per OGS player (one question "
                   "per colour, newest <= 20 rows by start_time; mean of the colours' log-probabilities, "
                   "renormalised); isotonic non-decreasing fit f of e on the OGS median rank (equal weights, PAV); "
                   "label = floor(f + 0.5); dropped iff f < -0.5 or f >= 9.5. Only OGS data; no inner data, no "
                   "validation label",
           "rule": {"question_max_rows": QUESTION_MAX, "newest": "start_time descending, ties: larger OGS game id",
                    "colour_mean": "arithmetic mean of log-probability vectors, then softmax renormalisation",
                    "levels": list(LEVELS), "isotonic": "PAV, equal weights, tied x pooled first",
                    "round": "floor(f + 0.5)", "keep": "-0.5 <= f < 9.5"},
           "inputs": {"prep_dir": rel(prep_dir), "prep_files_sha256": fsha,
                      "prep_record_rows": rows.record.get("rows"), "q09_map_sha256": map_sha, "seam_f_s25": seam_f},
           "players": len(res["players"]), "rows": int(rows.n), "questions_per_colour": res["questions_per_colour"],
           "question_rows": res["question_rows"],
           "dropped_out_of_range": int((~keep).sum()),
           "record_only": {"per_ogs_bin": per_bin(res), "f_knots": knots(res["ux"], res["fu"]),
                           "e_mean_all": float(res["e"].mean()),
                           "players_per_level": dict(zip(LEVELS, np.bincount(lab[keep], minlength=10).tolist()))},
           "labels": labels, "seconds": round(time.time() - t0, 1)}
    PR.write_json(out, rec)
    log(json.dumps({"players": rec["players"], "kept": len(labels), "per_level":
                    rec["record_only"]["players_per_level"], "seconds": rec["seconds"]}))
    log(f"-> {rel(out)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("align")
    p.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    return {"align": cmd_align}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
