"""PREREG_13 (competition 1) B-prime: the frozen B of PREREG_10 re-judged on the black-only view of the local
validation pool (docs/PREREG_13_PLAYER_B_BLACK.md sections 2-5).

    PY = .venv/Scripts/python.exe   (CPU; Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL
                                     thread variables)
    PY tools/p13_view.py apply [--rerun --reason T]  -> results/prereg13/player/view_report.json, view_06_top5.csv,
                                                        view_10b_top5.csv  (score_allowed 'pending_audit' or false)
    PY tools/p13_view.py apply --finalize             (after prescore_audit.json is committed: score_allowed only)
    PY tools/p13_view.py status                       (prints the state; writes nothing)

What apply does (in this order)
    1. prerequisites: the tools and PREREG_13 committed and clean; PREREG_10's frozen B (frozen_config.json,
       p1_lda.npz, val_report.json and its 225 validation P1 chunks) as tools/p10_apply.py checks them; the mock-path
       tools equal the ones frozen_config.json records; results/prereg10/player/apply_report.json committed.
    2. section 2.1: every game's colour from the committed prereg02 dumps, checked game by game against the two
       validation game files (a difference: exit 2).
    3. section 3: the identity view (every game, all 400 questions, sr06 strict=True) through the same code as the
       view. 06's Top-5 must equal results/prereg06/player/player_val_top5.csv, the sr06 means (6 decimals) and
       sr06_10B must equal PREREG_10's apply_report.json, V2 selected, 10B's Top-5 must equal
       results/prereg10/player/player_val_top5.csv and its pool gate -0.019891447034757093 (|diff| <= 1e-12).
       Anything else: exit 1 (a program error to fix, not a registered outcome); nothing is written.
    4. section 2.2-2.5: the view = candidate groups and question groups holding only the rows whose target player
       is black; the questions with >= 5 such games (n_view). Every pool statistic (IDF, z, CSLS, the LDA / S2 / JO
       colour terms, T_P1) is computed from the groups, so this equals deleting the white rows
       (tests/test_p13_view.py checks it byte for byte on a synthetic pool). 06 re-selects its combination on the
       view (sr06 with p06_style.sr06_rounds(strict=False)), 10B = fuse_10b(F_sel, z(T_P1), w*, k*).
    5. section 4: V1 structure (n_view = 263; every candidate >= 1 black game; P1 replay failures in the view
       <= 0.5 %; otherwise exit 2, nothing written), V2 sr(selected) - sr(R02) >= 0, V3 sr(10B) - sr(selected) >= 0.
    6. section 5.2: two 400-row files in the validation question order: the n_view rows from 06/view (A) and
       10B/view (B); the other rows copied verbatim from results/prereg06/player/player_val_top5.csv in both;
       check_submission on both; view_report.json with score_allowed 'pending_audit' (all conditions met; the audit
       comes next) or false with the reasons.

Discipline: imports only tools/p10_*, p06_*, p02_* (and the stdlib / numpy / scipy); the p06 data/local guard is on
(installed by p10_fuse); only the two validation game files are opened, inside p06_style.allow_local(); the
scoring tool is never run or imported; no torch / p1_lib / KataGo; no GPU.
Exit codes (as tools/p10_apply.py): 0 done (score_allowed 'pending_audit', finalized, or status); 1 registered stop
(identity view not reproduced; V2 or V3 not passed; format check failed; audit found an error); 2 refused or error
(including V1 and the colour check).
"""
import datetime
import hashlib
import os
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p10_apply as A  # noqa: E402  (imports p10_fuse, which installs the p06 guard)
import p10_fuse as F  # noqa: E402
from p10_fuse import (FAIL_MAX, Refused, Stop, commit_time, fuse_10b, git, path_of, project_p1, read_json,  # noqa
                      rel, require_committed, sha_lf, sha_raw, sim, st, tracked_clean, write_json, z_p1)

# ================================================================ constants (PREREG_13 rev 1)
PREREG13 = path_of("docs/PREREG_13_PLAYER_B_BLACK.md")
OUT13 = path_of("results/prereg13/player")
P10_OUT = F.OUT
P06_PLAYER = F.P06_PLAYER
P02_PLAYER = A.P02_PLAYER
P02_DUMPS = os.path.join(P02_PLAYER, "dumps")
FORMATS13 = {"view": "c10-prereg13 player view_report v1", "audit": "c10-prereg13 player prescore_audit v1"}
BLACK = "B"
MIN_VIEW_GAMES = 5                   # 2.3: the official minimum number of games per question
N_VIEW_EXPECTED = 263                # 2.3 / V1: counted while drafting; a different count stops (exit 2)
GATE_V2_MIN = 0.0                    # = tools/t06_player.GATE06_MIN (PREREG_06 2.7)
GATE_V3_MIN = 0.0                    # = PREREG_10 2.5
NONINF_PER_400 = -0.010              # 6.3: ci95 lower bound must exceed -0.010 * n_view / 400
REPRO_GATE = -0.019891447034757093   # section 3: PREREG_10's pool gate on the mixed-colour pool
REPRO_TOL = 1e-12
REPRO_SHA06_PREFIX = "094223f6"     # section 3: results/prereg06/player/player_val_top5.csv
REPRO_SHA10_PREFIX = "0efd160d"     # section 3: results/prereg10/player/player_val_top5.csv (apply_report.json)
REPRO_SELECTED = "V2"                # section 3: PREREG_10 selected V2 on the mixed-colour pool
OUT_A = "view_06_top5.csv"
OUT_B = "view_10b_top5.csv"
REPORT = "view_report.json"
AUDIT = "prescore_audit.json"
AUDIT_BOUND_OUTPUTS = (OUT_A, OUT_B, REPORT)
AUDIT_BOUND_INPUTS = ("p1_lda.npz", "frozen_config.json", "val_report.json")
P13_TOOL_FILES = A.APPLY_TOOL_FILES + ("tools/p13_view.py",)


def lf(b):
    return b.replace(b"\r\n", b"\n")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ================================================================ the pool (loaded once; views are row groups)
class Pool:
    """Every validation game once: the 06 store (MemStore, O_raw, CNN arrays), the frozen 06 LDAs, the P1
    projection with the frozen p1_lda, and each row's colour (target player's colour)."""


def load_pool(cand_path, q_path, p02_dir, p06_dir, vfiles, val_set, vr, model, smoke=False):
    import scipy.sparse as sp
    import p02_player_fuse as pf
    import p06_player as P6
    rep06 = read_json(os.path.join(p06_dir, "apply_report.json"))
    fz = read_json(os.path.join(p06_dir, "frozen_config.json"))
    try:
        with st.allow_local():
            d = pf.Data(cand_path, p02_dir, True, smoke, q_path=q_path)
            cand_rows = pf.read_candidates(cand_path)
            q_rows = pf.read_questions(q_path)
    except SystemExit as e:              # p02's Data reports a dump / file mismatch with SystemExit
        raise Refused(f"validation data: {e}")
    if any(d.shas.get(k) != rep06["inputs"].get(k) for k in d.shas):
        raise Refused("the validation game files or prereg02 dumps differ from the ones 06 applied to")
    keys, file_cols = A.val_keys(cand_rows, q_rows)
    if A.keys_sha(keys) != vr["keys_sha256"] or len(keys) != vr["games"]:
        raise Refused("validation game order differs from val_report.json")
    models = {cc: P6.load_lda(os.path.join(p06_dir, "lda_%s.npz" % cc)) for cc in "BW"}
    by_gid = {r[0]: r for r in cand_rows}
    games = [(g[1], g[0], g[2], by_gid[g[0]][3]) for g in d.cgames]
    qsgf = {(q, s): (cc, x) for q, g in q_rows for s, cc, x in g}
    games += [(g[0], "%s#%d" % (g[0], g[1]), g[2], qsgf[(g[0], g[1])][1]) for g in d.qgames]
    if [str(g[1]) for g in games] != keys:
        raise Refused("06's game order differs from the P1 extraction order")
    # ---- 2.1 colours from the committed prereg02 dumps, checked against the game files game by game
    dump_cols = dump_colours(p02_dir, d)
    if dump_cols != [str(c) for c in file_cols]:
        bad = [keys[i] for i, (x, y) in enumerate(zip(dump_cols, file_cols)) if x != y]
        raise Refused(f"colour of the prereg02 dumps differs from the game files for {len(bad)} games, "
                      f"e.g. {bad[:3]} (PREREG_13 2.1)")
    gc, gq = d.G("raw")
    p = Pool()
    p.d, p.P6, p.keys, p.nc = d, P6, keys, len(d.cgames)
    p.colour = np.array(dump_cols)
    p.ms = P6.MemStore(games, sp.vstack([gc, gq]).tocsr(), models)
    p.earr = (np.vstack([d.cemb, d.qemb]), np.concatenate([d.cnw, d.qnw]), np.concatenate([d.cok, d.qok]))
    p.models = models
    p.combos = sim.combo_list(fz["combinations"]["V1"])
    p.order = [n for n, _w in p.combos]
    p.kst = fz["k_star"]
    p.rep06 = rep06
    X, usable, failed, have, cnt = F.load_side0(vfiles, len(keys), val_set)
    if not have.all():
        raise Refused("validation games without a chunk")
    Z = np.zeros((len(keys), st.LDA_DIM))
    rows = np.flatnonzero(usable)
    Z[rows] = project_p1(X, model, rows)
    p.Z, p.usable, p.failed, p.cnt = Z, usable, failed, cnt
    p.shas = dict(d.shas)
    return p


def dump_colours(p02_dir, d):
    """Colour of every validation game (candidate games in file order, then question slots) from the prereg02
    dumps: cand_games.npz 'color' by game_id, query_games.npz 'color' by (question_id, slot)."""
    with np.load(os.path.join(p02_dir, "dumps", "cand_games.npz"), allow_pickle=False) as z:
        cc = dict(zip([str(x) for x in z["game_id"].tolist()], [str(x) for x in z["color"].tolist()]))
    with np.load(os.path.join(p02_dir, "dumps", "query_games.npz"), allow_pickle=False) as z:
        qc = dict(zip(zip([str(x) for x in z["question_id"].tolist()], [int(s) for s in z["slot"].tolist()]),
                      [str(x) for x in z["color"].tolist()]))
    out = [cc.get(str(g[0])) for g in d.cgames]
    out += [qc.get((str(g[0]), int(g[1]))) for g in d.qgames]
    return out


# ================================================================ views
class View:
    pass


def identity_view(p):
    """Section 3: every game, all questions, sr06 strict=True (= PREREG_10's apply)."""
    v = View()
    v.name = "identity"
    v.cand = {c: list(p.d.c_by_player[c]) for c in p.d.cand_ids}
    v.qids = list(p.d.qids)
    v.q = {q: [p.nc + i for i in p.d.q_by_id[q]] for q in v.qids}
    v.strict = True
    return v


def black_view(p, min_games=MIN_VIEW_GAMES):
    """Section 2: candidate groups = the candidate's rows whose target player is black (all 400 candidates kept);
    question groups = the question's black rows in slot order; the questions with >= min_games such rows."""
    v = View()
    v.name = "black_only"
    v.cand = {c: [r for r in p.d.c_by_player[c] if p.colour[r] == BLACK] for c in p.d.cand_ids}
    v.qblack = {q: [p.nc + i for i in p.d.q_by_id[q] if p.colour[p.nc + i] == BLACK] for q in p.d.qids}
    v.qids = [q for q in p.d.qids if len(v.qblack[q]) >= min_games]
    v.q = {q: v.qblack[q] for q in v.qids}
    v.strict = False
    return v


def sr06_view(P6, ms, cand_ids, c_by_pid, ks, earr, models, combos, kst, Z, usable, w_star, strict):
    """p10_apply.sr06_with_p1 with sr06_rounds(strict=strict) (PREREG_13 2.4 step 1; p10_apply.py is not changed).
    A pseudo question that is empty (strict=False, a short part) is not asked (as p06_player.sr06_apply's `if q`).
    -> ({combo: mean, 'R02': mean}, {combo: 10B mean})."""
    cg = {p: [str(ms.gid[r]) for r in c_by_pid[p]] for p in cand_ids}
    row_of = {(p, str(ms.gid[r])): r for p in cand_ids for r in c_by_pid[p]}
    rr = st.sr06_rounds(cg, ks, strict=strict)
    pts = {n: [] for n, _w in combos}
    pts["R02"] = []
    p10 = {n: [] for n, _w in combos}
    for r, items in enumerate(rr):
        cgr = [[row_of[(p, g)] for g in refs] for p, _q, refs in items]
        qsel = [(i, [row_of[(p, g)] for g in q]) for i, (p, q, _refs) in enumerate(items) if q]
        qi = np.array([i for i, _r in qsel], np.int64)
        qg = [rws for _i, rws in qsel]
        T = P6.apply_terms(ms, models, cgr, qg, (earr, cgr, earr, qg))
        zT = z_p1(Z, cgr, qg, usable)
        for n, w in combos:
            Fm = sim.combo_F(T, "E0", w)
            pts[n] += st.points(st.ranks(st.csls(Fm, kst), qi)).tolist()
            p10[n] += st.points(st.ranks(fuse_10b(Fm, zT, w_star, kst), qi)).tolist()
        pts["R02"] += st.points(st.ranks(sim.combo_F(T, "E0", None), qi)).tolist()
    return ({n: float(np.mean(v)) for n, v in pts.items()}, {n: float(np.mean(v)) for n, v in p10.items()})


def run_view(p, v, w_star):
    """06 (re-selected on the view) and 10B on one view -> dict (sr06, sr06_10B, selected, the two Top-5 lists)."""
    cand_ids = p.d.cand_ids
    cgroups = [v.cand[c] for c in cand_ids]
    qgroups = [v.q[q] for q in v.qids]
    ks = [len(g) for g in qgroups]
    srm, sr10 = sr06_view(p.P6, p.ms, cand_ids, v.cand, ks, p.earr, p.models, p.combos, p.kst, p.Z, p.usable,
                          w_star, v.strict)
    chosen = p.P6.choose_combo({n: srm[n] for n in p.order}, p.order)
    T = p.P6.apply_terms(p.ms, p.models, cgroups, qgroups, (p.earr, cgroups, p.earr, qgroups))
    Fsel = sim.combo_F(T, "E0", dict(p.combos)[chosen])
    top06 = st.top5(st.csls(Fsel, p.kst), cand_ids)
    zT = z_p1(p.Z, cgroups, qgroups, p.usable)
    top10 = st.top5(fuse_10b(Fsel, zT, w_star, p.kst), cand_ids)
    return {"sr06": srm, "sr06_10B": sr10, "selected": chosen,
            "gate_v2": srm[chosen] - srm["R02"], "gate_v3": sr10[chosen] - srm[chosen],
            "top06": top06, "top10": top10, "qids": list(v.qids),
            "changed_top1": int(sum(1 for x, y in zip(top06, top10) if x[0] != y[0]))}


def top5_bytes(P6, qids, tops):
    """The bytes p06_player.write_top5 writes (written to a file in a fresh system temporary folder, read back,
    removed: nothing is left in the output folder)."""
    d = tempfile.mkdtemp(prefix="c10p13_")
    try:
        t = os.path.join(d, "top5.csv")
        P6.write_top5(t, qids, tops)
        with open(t, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def compose_400(P6, all_qids, view_res_qids, view_tops, ref06_path):
    """5.2: the 400-row file in the question file's order; view rows from view_tops, the other rows verbatim from
    06's committed file. -> bytes."""
    with open(ref06_path, "rb") as f:
        ref = lf(f.read()).decode("utf-8").split("\n")
    if ref[-1] != "":
        raise Refused("06's Top-5 file does not end with a newline")
    ref = ref[:-1]
    header, rows = ref[0], ref[1:]
    by_q = {}
    for ln in rows:
        by_q[ln.split(",", 1)[0]] = ln
    if len(by_q) != len(rows) or sorted(by_q) != sorted(all_qids):
        raise Refused("06's Top-5 file does not cover the validation questions once each")
    vb = top5_bytes(P6, view_res_qids, view_tops).decode("utf-8").split("\n")
    if vb[0] != header:
        raise Refused("06's Top-5 header differs from p06_player.write_top5's")
    vrow = {ln.split(",", 1)[0]: ln for ln in vb[1:-1]}
    out = [header] + [vrow[q] if q in vrow else by_q[q] for q in all_qids]
    return ("\n".join(out) + "\n").encode("utf-8")


def nonview_identical(a_bytes, b_bytes, ref06_path, view_qids):
    """5.3: the rows outside the view are the same in A, B and 06 (line by line, LF)."""
    with open(ref06_path, "rb") as f:
        ref = lf(f.read()).decode("utf-8").split("\n")[1:-1]
    a = a_bytes.decode("utf-8").split("\n")[1:-1]
    b = b_bytes.decode("utf-8").split("\n")[1:-1]
    vs = set(view_qids)
    rq = {ln.split(",", 1)[0]: ln for ln in ref}
    pairs = [(x, y) for x, y in zip(a, b) if x.split(",", 1)[0] not in vs]
    return (len(a) == len(b) == len(ref) and [x.split(",", 1)[0] for x in a] == [x.split(",", 1)[0] for x in b]
            and all(x == y == rq[x.split(",", 1)[0]] for x, y in pairs)
            and len(pairs) == len(ref) - len(vs))


# ================================================================ section 3: the identity reproduction
def reproduce(p, w_star, p10_rep, ref06_path, ref10_path):
    """-> (ok, record). Every condition of section 3."""
    res = run_view(p, identity_view(p), w_star)
    b06 = top5_bytes(p.P6, res["qids"], res["top06"])
    b10 = top5_bytes(p.P6, res["qids"], res["top10"])
    with open(ref06_path, "rb") as f:
        r06 = lf(f.read())
    with open(ref10_path, "rb") as f:
        r10 = lf(f.read())
    sr_ok = (all(round(res["sr06"][n], 6) == p10_rep["sr06"][n] for n in p10_rep["sr06"])
             and set(p10_rep["sr06"]) == set(res["sr06"]))
    sr10_ok = (all(round(res["sr06_10B"][n], 6) == p10_rep["sr06_10B"][n] for n in p10_rep["sr06_10B"])
               and set(p10_rep["sr06_10B"]) == set(res["sr06_10B"]))
    rec = {"top5_06_identical": b06 == r06, "top5_06_sha256": sha_bytes(b06),
           "top5_06_reference": rel(ref06_path),
           "top5_06_reference_sha256": sha_bytes(r06),
           "top5_06_reference_sha256_in_p10_report": p10_rep["p06_reproduction"]["reference_sha256"],
           "sr06": {n: round(x, 6) for n, x in res["sr06"].items()}, "sr06_equal": sr_ok,
           "sr06_10B": {n: round(x, 6) for n, x in res["sr06_10B"].items()}, "sr06_10B_equal": sr10_ok,
           "selected": res["selected"], "selected_equal": res["selected"] == p10_rep["selected"] == REPRO_SELECTED,
           "top5_10b_identical": b10 == r10, "top5_10b_sha256": sha_bytes(b10),
           "top5_10b_reference": rel(ref10_path), "top5_10b_reference_sha256": sha_bytes(r10),
           "top5_10b_reference_sha256_in_p10_report": p10_rep["outputs_sha256"]["player_val_top5.csv"],
           "pool_gate": res["gate_v3"], "pool_gate_reference": REPRO_GATE,
           "pool_gate_equal": abs(res["gate_v3"] - REPRO_GATE) <= REPRO_TOL
           and abs(res["gate_v3"] - p10_rep["pool_gate"]["sr_10B_minus_selected"]) <= REPRO_TOL,
           "changed_top1": res["changed_top1"]}
    rec["references_as_recorded"] = (
        rec["top5_06_reference_sha256"] == rec["top5_06_reference_sha256_in_p10_report"]
        and rec["top5_10b_reference_sha256"] == rec["top5_10b_reference_sha256_in_p10_report"]
        and rec["top5_06_reference_sha256"].startswith(REPRO_SHA06_PREFIX)
        and rec["top5_10b_reference_sha256"].startswith(REPRO_SHA10_PREFIX))
    ok = all(rec[k] for k in ("top5_06_identical", "sr06_equal", "sr06_10B_equal", "selected_equal",
                              "top5_10b_identical", "pool_gate_equal", "references_as_recorded"))
    rec["reproduced"] = ok
    return ok, rec


# ================================================================ the view (sections 2, 4, 5)
def view_structure(p, v):
    nb_c = {c: len(v.cand[c]) for c in p.d.cand_ids}
    nb_q = {q: len(v.qblack[q]) for q in p.d.qids}
    rows = sorted({r for c in p.d.cand_ids for r in v.cand[c]} | {r for q in v.qids for r in v.q[q]})
    rows = np.asarray(rows, np.int64)
    n_fail = int(p.failed[rows].sum())
    cvals = np.array(list(nb_c.values()))
    qvals = np.array([nb_q[q] for q in v.qids]) if v.qids else np.zeros(0)
    allq = np.array(list(nb_q.values()))
    return {"n_view": len(v.qids), "view_question_ids": list(v.qids),
            "candidate_black_games": nb_c, "question_black_games": nb_q,
            "candidates": len(nb_c), "candidates_without_black": int((cvals == 0).sum()),
            "candidate_black_games_summary": {"min": int(cvals.min()), "p5": float(np.percentile(cvals, 5)),
                                              "median": float(np.median(cvals)), "max": int(cvals.max()),
                                              "under_10": int((cvals < 10).sum()),
                                              "under_25": int((cvals < 25).sum()),
                                              "total": int(cvals.sum())},
            "view_question_black_games_summary": {"min": int(qvals.min()) if len(qvals) else None,
                                                  "max": int(qvals.max()) if len(qvals) else None,
                                                  "total": int(qvals.sum())},
            "questions_by_black_games": {"0": int((allq == 0).sum()),
                                         "1-4": int(((allq >= 1) & (allq < MIN_VIEW_GAMES)).sum()),
                                         ">=5": int((allq >= MIN_VIEW_GAMES).sum())},
            "p1": {"view_games": int(len(rows)), "failed": n_fail,
                   "failed_fraction": n_fail / len(rows) if len(rows) else 0.0,
                   "no_target_moves": int(((p.cnt[rows] == 0) & ~p.failed[rows]).sum()),
                   "usable": int(p.usable[rows].sum())}}


def v1_problems(struct, n_expected=N_VIEW_EXPECTED):
    probs = []
    if struct["n_view"] != n_expected:
        probs.append(f"n_view {struct['n_view']} != {n_expected} (PREREG_13 2.3 / V1)")
    if struct["candidates_without_black"]:
        probs.append(f"{struct['candidates_without_black']} candidates without a black game (V1)")
    if struct["p1"]["failed_fraction"] > FAIL_MAX:
        probs.append(f"P1 failed games in the view {struct['p1']['failed_fraction']:.4%} > 0.5 % (V1)")
    return probs


def view_decision(repro_ok, v1_ok, v2_ok, v3_ok, fmt_ok, nonview_ok):
    """score_allowed of 5.3 before the audit -> (False / 'pending_audit', reasons). apply never writes True."""
    reasons = []
    if not repro_ok:
        reasons.append("section 3 identity view not reproduced")
    if not v1_ok:
        reasons.append("V1 structure not passed")
    if not v2_ok:
        reasons.append("V2 not passed: sr(06 selected) - sr(R02) < 0 on the view; B-prime not used (not submitted)")
    if not v3_ok:
        reasons.append("V3 not passed: sr(10B) - sr(06 selected) < 0 on the view; B-prime not used (not submitted)")
    if not fmt_ok:
        reasons.append("check_submission did not pass")
    if not nonview_ok:
        reasons.append("the rows outside the view are not identical in A, B and 06")
    if reasons:
        return False, reasons
    return "pending_audit", ["prescore_audit.json of these outputs not committed yet (PREREG_13 5.3); then "
                             "apply --finalize"]


def view_core(p, w_star, p10_rep, ref06_path, ref10_path, out_dir, q_path, cand_path, n_expected=N_VIEW_EXPECTED,
              check=True):
    """Sections 3, 2, 4, 5.2 on a loaded pool -> (report dict, A bytes, B bytes). Raises Stop when section 3 is not
    reproduced, Refused when V1 fails (nothing written in both cases). Writes the two files into out_dir."""
    ok, repro = reproduce(p, w_star, p10_rep, ref06_path, ref10_path)
    if not ok:
        bad = [k for k in ("top5_06_identical", "sr06_equal", "sr06_10B_equal", "selected_equal",
                           "top5_10b_identical", "pool_gate_equal", "references_as_recorded") if not repro[k]]
        raise Stop(f"section 3 identity view not reproduced: {bad} (fix the program; not a registered outcome)")
    v = black_view(p)
    struct = view_structure(p, v)
    probs = v1_problems(struct, n_expected)
    if probs:
        raise Refused(f"V1: {probs} (stop and check; PREREG_13 section 4)")
    res = run_view(p, v, w_star)
    a = compose_400(p.P6, p.d.qids, res["qids"], res["top06"], ref06_path)
    b = compose_400(p.P6, p.d.qids, res["qids"], res["top10"], ref06_path)
    nonview_ok = nonview_identical(a, b, ref06_path, res["qids"])
    os.makedirs(out_dir, exist_ok=True)
    pa, pb = os.path.join(out_dir, OUT_A), os.path.join(out_dir, OUT_B)
    for path, data in ((pa, a), (pb, b)):
        with open(path, "wb") as f:
            f.write(data)
    if check:
        fa = A.run_check_submission(pa, q_path, cand_path)
        fb = A.run_check_submission(pb, q_path, cand_path)
    else:
        fa = fb = {"returncode": 0, "stdout": "not run (test)", "stderr": ""}
    v2_ok = res["gate_v2"] >= GATE_V2_MIN
    v3_ok = res["gate_v3"] >= GATE_V3_MIN
    fmt_ok = fa["returncode"] == 0 and fb["returncode"] == 0
    allowed, reasons = view_decision(True, True, v2_ok, v3_ok, fmt_ok, nonview_ok)
    n_view = struct["n_view"]
    rep = {"format": FORMATS13["view"],
           "what": "PREREG_13 sections 2-5: frozen B (PREREG_10) re-judged on the black-only view of the local "
                   "validation pool",
           "view": dict(struct, definition={
               "colour_source": "prereg02 dumps cand_games.npz / query_games.npz 'color' (checked equal to the "
                                "validation game files game by game)",
               "candidates": "all 400; only the games whose target player is black",
               "questions": "only the black games, slot order; questions with >= %d such games" % MIN_VIEW_GAMES,
               "sr06_rounds_strict": False}),
           "reproduction": repro,
           "sr06": {n: round(x, 6) for n, x in res["sr06"].items()},
           "sr06_10B": {n: round(x, 6) for n, x in res["sr06_10B"].items()},
           "selected": res["selected"],
           "gates": {"V1": {"passed": True, "n_view": n_view, "n_view_expected": n_expected,
                            "candidates_without_black": struct["candidates_without_black"],
                            "p1_failed_fraction": struct["p1"]["failed_fraction"], "p1_failed_max": FAIL_MAX},
                     "V2": {"sr_selected_minus_R02": res["gate_v2"], "min": GATE_V2_MIN, "passed": bool(v2_ok)},
                     "V3": {"sr_10B_minus_selected": res["gate_v3"], "min": GATE_V3_MIN, "passed": bool(v3_ok)}},
           "w_star": w_star, "k_star": p.kst,
           "changed_top1_06_vs_10B_in_view": res["changed_top1"],
           "margin": {"n_view": n_view, "mean_diff_min": 0.0,
                      "ci95_lower_must_exceed": NONINF_PER_400 * n_view / 400,
                      "rule": "PREREG_13 6.3: mean_diff >= 0 and ci95[0] > -0.010 * n_view / 400"},
           "nonview_rows_identical": bool(nonview_ok), "nonview_rows": len(p.d.qids) - n_view,
           "format_check": {OUT_A: fa, OUT_B: fb},
           "outputs_sha256": {OUT_A: sha_bytes(a), OUT_B: sha_bytes(b)},
           "inputs_sha256": dict(p.shas), "score_allowed": allowed, "score_blocked_reasons": reasons}
    return rep, a, b


# ================================================================ audit / finalize
def audit_binding(audit, rep, rep_sha, audit_commit, rep_commit):
    probs = []
    if audit.get("format") != FORMATS13["audit"]:
        probs.append(f"audit format {audit.get('format')!r} != {FORMATS13['audit']!r}")
    if not isinstance(audit.get("program_errors_affecting_view"), list):
        probs.append("audit field program_errors_affecting_view is not a list")
    if audit.get("independent_recompute_identical") is not True:
        probs.append("audit does not say the independent recompute is identical")
    outs = audit.get("outputs_sha256") or {}
    for n_ in (OUT_A, OUT_B):
        if outs.get(n_) != (rep.get("outputs_sha256") or {}).get(n_):
            probs.append(f"audit outputs_sha256[{n_}] differs from view_report.json")
    if outs.get(REPORT) != rep_sha:
        probs.append("audit outputs_sha256[view_report.json] differs from the committed view_report.json")
    for n_ in AUDIT_BOUND_INPUTS:
        if (audit.get("inputs_sha256") or {}).get(n_) != (rep.get("inputs_sha256") or {}).get(n_):
            probs.append(f"audit inputs_sha256[{n_}] differs from view_report.json")
    if (not rep_commit or not audit_commit
            or datetime.datetime.fromisoformat(audit_commit) <= datetime.datetime.fromisoformat(rep_commit)):
        probs.append(f"audit commit {audit_commit} is not later than view_report.json's {rep_commit}")
    return probs


def audit_state(out_dir, rep=None):
    p = os.path.join(out_dir, AUDIT)
    if not os.path.exists(p):
        return "missing", None
    if tracked_clean([p]):
        return "missing", {"note": "prescore_audit.json exists but is not committed"}
    au = read_json(p)
    rec = {"sha256": sha_lf(p), "commit": commit_time(p)}
    if rep is not None:
        rp = os.path.join(out_dir, REPORT)
        probs = audit_binding(au, rep, sha_lf(rp), rec["commit"], commit_time(rp))
        if probs:
            rec["binding_problems"] = probs
            return "unbound", rec
    return ("bug" if au.get("program_errors_affecting_view") else "ok"), rec


def finalize_decision(rep, audit, audit_rec):
    return A.finalize_decision(rep, audit, audit_rec)


# ================================================================ commands
def apply_prerequisites13():
    """-> (PREREG_10 prerequisites..., PREREG_10 apply report, commits)."""
    fc, commits, model, vr, vfiles = A.apply_prerequisites()
    rp10 = os.path.join(P10_OUT, "apply_report.json")
    commits["p10_apply_report"] = require_committed([rp10, os.path.join(P10_OUT, "player_val_top5.csv")],
                                                    "PREREG_10 apply outputs")
    commits["prereg13"] = require_committed([PREREG13], "PREREG_13")
    rc, out = git(["log", "--format=%H %cI", "-S", "| rev 1 |", "--", rel(PREREG13)])
    if rc != 0 or not out.strip():
        raise Refused("PREREG_13 rev 1 is not in any commit (K1-13)")
    commits["prereg13"]["rev1"] = out.strip().splitlines()[-1]
    rc, out = git(["log", "--format=%H %cI", "-S", "| rev 2 |", "--", rel(PREREG13)])
    if rc != 0 or not out.strip():
        raise Refused("PREREG_13 rev 2 (the implementation's differences, section 14 step 2) is not committed")
    commits["prereg13"]["rev2"] = out.strip().splitlines()[-1]
    return fc, commits, model, vr, vfiles, read_json(rp10)


def cmd_apply(a):
    if a.finalize:
        return apply_finalize(a)
    t0 = time.time()
    F.blas_env_check(False)
    ts = F.tool_state(P13_TOOL_FILES)
    F.require_tools(ts)
    paths = [os.path.join(OUT13, n_) for n_ in (REPORT, OUT_A, OUT_B)]
    rerun = None
    if any(os.path.exists(x) for x in paths):
        if not (a.rerun and a.reason):
            raise Refused("apply outputs exist; rerun only with --rerun --reason TEXT")
        if any(git(["ls-files", "--error-unmatch", "--", rel(x)])[0] == 0 for x in paths):
            raise Refused("committed apply outputs are never overwritten")
        rerun = {"reason": a.reason}
    fc, commits, model, vr, vfiles, p10_rep = apply_prerequisites13()
    diff = F.tools_differ(ts, fc.get("tools"))
    if diff:
        raise Refused(f"mock-path tools changed since the freeze: {diff} (T_P1 / fusion must be the frozen code)")
    p = load_pool(A.VAL_CANDIDATES, A.VAL_QUESTIONS, P02_PLAYER, P06_PLAYER, vfiles, A.VAL_SET, vr, model)
    rep, _a, _b = view_core(p, float(fc["w_star"]), p10_rep, os.path.join(P06_PLAYER, "player_val_top5.csv"),
                            os.path.join(P10_OUT, "player_val_top5.csv"), OUT13, A.VAL_QUESTIONS, A.VAL_CANDIDATES)
    rep["inputs_sha256"].update({"p1_lda.npz": sha_raw(os.path.join(P10_OUT, "p1_lda.npz")),
                                 "frozen_config.json": sha_lf(os.path.join(P10_OUT, "frozen_config.json")),
                                 "val_report.json": sha_lf(A.VAL_REPORT),
                                 "p10_apply_report.json": sha_lf(os.path.join(P10_OUT, "apply_report.json"))})
    rep.update({"commits": commits, "rerun": rerun})
    rep.update(F.meta(A._real_ctx(), t0, ts))
    write_json(os.path.join(OUT13, REPORT), rep)
    g = rep["gates"]
    print(f"apply: identity reproduced {rep['reproduction']['reproduced']}, n_view {rep['view']['n_view']}, "
          f"selected {rep['selected']}, V2 {g['V2']['sr_selected_minus_R02']:+.6f}, "
          f"V3 {g['V3']['sr_10B_minus_selected']:+.6f}, score_allowed {rep['score_allowed']} "
          f"{rep['score_blocked_reasons']}")
    return 0 if rep["score_allowed"] == "pending_audit" else 1


def apply_finalize(_a):
    paths = [os.path.join(OUT13, n_) for n_ in (REPORT, OUT_A, OUT_B)]
    require_committed(paths, "apply outputs")
    rp = paths[0]
    rep = read_json(rp)
    if rep.get("format") != FORMATS13["view"]:
        raise Refused("view_report.json is not a c10-prereg13 view report")
    for n_, sha in rep["outputs_sha256"].items():
        with open(os.path.join(OUT13, n_), "rb") as f:
            if sha_bytes(f.read()) != sha:
                raise Refused(f"{n_} changed since apply")
    audit, audit_rec = audit_state(OUT13, rep)
    allowed, reasons = finalize_decision(rep, audit, audit_rec)
    if audit_rec is not None:
        audit_rec["view_report_sha256_at_audit"] = sha_lf(rp)
    rep["score_allowed"], rep["score_blocked_reasons"] = allowed, reasons
    rep["audit"] = audit_rec
    rep["finalized"] = F.now_iso()
    rep["finalized_by_tool"] = F.tool_state(P13_TOOL_FILES)["files_sha256"]
    write_json(rp, rep)
    print(f"score_allowed: {allowed}")
    return 0 if allowed is True else 1


def cmd_status(_a):
    rp = os.path.join(OUT13, REPORT)
    if not os.path.exists(rp):
        print("PREREG_13: apply not run (no view_report.json)")
        return 0
    rep = read_json(rp)
    committed = not tracked_clean([rp])
    audit, rec = audit_state(OUT13, rep)
    g = rep.get("gates") or {}
    print(f"view_report.json: committed {committed}; n_view {(rep.get('view') or {}).get('n_view')}; "
          f"selected {rep.get('selected')}; "
          f"V2 {(g.get('V2') or {}).get('passed')}; V3 {(g.get('V3') or {}).get('passed')}; "
          f"score_allowed {rep.get('score_allowed')!r}; audit {audit}"
          + (f" {rec.get('binding_problems')}" if rec and rec.get("binding_problems") else ""))
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    sub.add_parser("status")
    a = ap.parse_args(argv)
    return A.run(cmd_apply if a.cmd == "apply" else cmd_status, a)


if __name__ == "__main__":
    sys.exit(main())
