"""PREREG_04 competition 2: R20 plus opponent-side evidence, fused in the generalised p02 calibration
(PREREG_04 rev 1 sections 2.1-2.6; results/prereg04/design/plan.md 2.1-2.3, 5 and 7).

    PY = .venv/Scripts/python.exe
    PY tools/p04_rank_fuse.py check      [--out-dir O] [--design-o4]   -> O/check_report.json
    PY tools/p04_rank_fuse.py selfcheck  [--out-dir O] [--design-o4]   -> O/selfcheck.json
    PY tools/p04_rank_fuse.py tune       [--out-dir O] [--design-o4]   -> O/tune_report.json, O/frozen_config.json
    PY tools/p04_rank_fuse.py apply                                    -> results/prereg04/rank/rank_val_probs.csv,
                                                                          results/prereg04/rank/apply_report.json
    then: PY tools/decide.py results/prereg04/rank/rank_val_probs.csv results/prereg04/rank
          PY tools/check_submission.py rank results/prereg04/rank/pred_B.csv
              --questions data/local/rank_val_questions.csv
    smoke (synthetic data only; every input path given, output outside the repository):
    PY tools/p04_rank_fuse.py <cmd> --smoke --p02-dir D --p02-local-dir L --p03-dir P --opp-dir X --split S --out-dir O
                                     [--questions Q]

Order (PREREG_04 9): check -> commit -> selfcheck -> commit -> tune -> commit frozen_config.json -> (p04_opp.py val)
-> apply -> decide.py -> check_submission.py (with the question file) -> commit -> H4R only when
apply_report.json says score_allowed. Real mode writes only into results/prereg04/rank. An --out-dir outside the
repository gives a *preview* of check / selfcheck / tune (real inputs, flagged "preview": true; never accepted by a
later real step). apply has no preview: it writes results/prereg04/rank only (or a --smoke out-dir).

--design-o4 (preview only; this build's verification): the O4 / T4 per-game vectors of the inner games come from the
design-phase file data/prereg04/design_scratch/proxy_inner_all.npz (SHA-256 pinned below and in
results/prereg04/design/scratch_manifest.json) instead of p04_opp.py's registered output. Every report is flagged
"design_input": true and "preview": true; the O4 count / dupcheck checks are reported as not run; tune never writes a
real frozen config from it and apply refuses it. No validation O4 exists in this mode.

Sources (per game, log-probabilities over the 10 levels, LEVELS order):
    C5   prereg02 epoch 9, 8 symmetries (R20's source)          p03 combo_logprobs(('C5',))
    OP   opening term (prereg02 opening_{inner,val}.npz)        question level: mean over the games that have one
    O4   opponent moves under the target's level tables         log_softmax(sum over the opponent's moves of log P_k)
    T4   target moves (reference only, colour check)            log_softmax(sum over the target's moves of log P_k)
    SWA  prereg03 arm A, original orientation, colour swapped   log_softmax(inner/val_p03a_s0_swap logits)
    SWB  prereg03 arm B, likewise                               log_softmax(inner/val_p03b_s0_swap logits)
    A0   prereg03 arm A symmetry 0, correct colour (reference)  log_softmax(inner/val_p03a_d4[:, 0])
    B0   prereg03 arm B likewise (reference)
    C6A  prereg03 arm A, 8 symmetries (record-only combination Q02+C6A)
Per question (n = parsed games): m_C5 = mean of l_C5; E_j = mean over the parsed games of l_j (0 without a game);
    p = softmax( e^a (n/12)^gamma (m_C5 + sum_j w_j E_j + b) ),  E = [OP, O4, SWA, SWB] (this order), b = B beta,
    a in (-4, 4), gamma in (-1, 2), w in (0, 4), starts a = 0 and 1, penalty 0.01 ||beta||^2, maxiter 500
    (gen_obj / gen_fit / gen_logits / gen_oof ported verbatim from results/prereg04/design/evidence/B_proxy/fuse.py).
Combinations (fixed order = tie priority): Q00 = R20 (C5 + OP), Q01 = R20 + O4, Q02 = R20 + O4 + SWA + SWB.
    Q00 runs through prereg02's own fit (p02_rank_fuse via p03_rank_fuse.run_oof / refit_all / val_probs) so it is R20
    bit for bit; Q01 / Q02 run through the generalised fit. The generalised fit of Q00 is a recorded seam (selfcheck
    requires the same rule-B decision on every inner eval question and the same OOF rank-balanced mean (<= 1e-12);
    the probability, fold-parameter and refit-parameter differences are recorded only): its refit on all fit
    questions differs from R20's frozen parameters by ~1.4e-12 (summation order (m + Bb) + w m_op vs (m + w m_op) + Bb),
    above the registered 1e-12, which is why Q00 is not taken from the generalised path (reported as a deviation).

check      1 structure: O4's key arrays equal prereg02's inner_keys (rank, game_id, player_id, colour), finite, 73,556
             rows; the pinned O4 numbers (count_report.json: 720,307 games, 79,985,811 target / 79,985,352 opponent
             moves, 0 anomalies, 557,928 cells; data/prereg04/counts_fit.npz equal array by array to the design counts;
             the inner O4 / T4 vectors within 1e-12 of proxy_inner_all.npz); SWA / SWB: key arrays equal prereg02's, the
             four swapped dumps' SHA-256 equal PREREG_04 1.3, weights equal the arm's d4 dumps and prereg03's check
             record,
             finite inner logits, validation NaN rows equal C5's; the A0 / B0 d4 dumps equal prereg03's check record.
             Any failure is a stop (status 'stop', exit 1).
           2 colour semantics (per opponent-side source against its target-side reference: O4 / T4, SWA / A0, SWB / B0):
             dkappa = kappa(reference) - kappa(source), kappa as PREREG_03 2.2 item 4, on the PREREG_03 rev 3
             single-colour inner pseudo-questions (120,544, SHA-256 45d0f14a...) with a PlayerBootstrap 95 % CI; usable
             only when the CI lower bound > 0 and (SWA, SWB) the validation single-colour sub-question dkappa > 0.
             O4's validation dkappa is computed by apply (needs the validation O4). A failing source is dropped with
             every combination that uses it (not a stop).
           3 dupcheck: p04_opp.py's report must exist; duplicated validation slots >= 1 % -> stop.
           4 selftest (real mode, not --design-o4): p04_opp.py's selftest.json (committed with the other p04_opp.py
             reports) must say passed, no problems, not a preview (plan.md 5: 0 capture / 0 feature mismatches).
           Also a stop: a validation O4 file present while frozen_config.json is not committed and unchanged
           (PREREG_04 1.2).
selfcheck  Q00 (prereg02's path) reproduces R20: the full inner OOF record of results/prereg02/rank/tune_report.json
           (0.5854702521055817, every level / bin / distribution / SE, every fold's parameters and loss), the refit
           vs frozen_config.json, the validation probabilities vs rank_val_probs.csv and pred_B.csv (all <= 1e-12, rule
           B identical). The generalised fit of Q00 gives the same OOF decisions (seam). Q01 and Q02 (generalised fit)
           reproduce 0.6042484782462081 and 0.6101211982700248 (<= 1e-12; a check of the port, not a choice).
tune       Q00-Q02 in the fixed order (dropped when check dropped a source), 5-fold OOF on prereg02's pseudo-question
           rounds; selection = best OOF rank-balanced mean (rule B), anything within 0.003 -> the earliest. Gates on the
           selected vs Q00: G1 >= +0.010; G2 paired PlayerBootstrap 95 % CI lower bound > 0; G3 mean per-question
           difference >= 0; G4 single-colour pseudo-questions: PlayerBootstrap lower bound > 0 and black and white point
           estimates > 0. Recorded only: O4 alone, R20+SWA+SWB, Q02+C6A, the no-opening versions, per level / bin /
           single colour (all, B, W, per k), se_val_record. Refit of the selected on all fit questions.
           frozen_config.json
           holds the selection, fold and refit parameters, source SHA-256s, gates, the single-colour kappa references
           (test day) and FAIL_OPENING = "04 does not play; test02's rule applies (R10)". Inner files only.
apply      Needs frozen_config.json committed and equal to HEAD (not preview, not design), the tools tune recorded, the
           inputs check recorded; check_report.json committed, unchanged and the one tune recorded (SHA-256 in
           frozen_config.json check.check_report_sha256; a passed real check). If the selection contains O4:
           results/prereg04/opp/val_o4.npz (p04_opp.py val) with key arrays equal to val_keys, replay_failed false
           everywhere and finite values on every row C5 parsed (rows C5 did not parse are ignored, set to 0; no NaN is
           required there), and val_report.json made from this committed frozen_config.json (its SHA-256), by the
           p04_opp.py tune recorded and with the rate tables tune froze (tables_sha256); O4's validation single-colour
           sub-question dkappa (T4 - O4) must be > 0, else score_allowed is false. Writes rank_val_probs.csv (prereg01
           layout, probabilities as repr, blank when n_parsed = 0) and apply_report.json (score_allowed,
           score_blocked_reasons).
No validation label file is ever read (every path goes through refuse_label_file); tools/score.py is never run.
Every command first compares tools/p02_*.py, p03_rank_fuse.py, c10data.py and decide.py with their HEAD blobs (SHA-256,
CRLF folded to LF); real mode also with the hashes prereg02's and prereg03's frozen configs recorded.
"""

import argparse
import csv
import glob
import math
import os
import sys
import time

import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import decide  # noqa: E402
import p02_rank_fuse as rf  # noqa: E402
import p03_rank_fuse as p3  # noqa: E402
from p02_rank_fuse import (BOOT_SEED, FLOAT_EPS, GATE, N_BOOT, N_LEVELS, PROB_COLUMNS, SELECT_TOL,  # noqa: E402
                           PlayerBootstrap, basis, load_dump, load_val_keys, log_softmax, metrics, params_dict,
                           pool_matrix, pooled, read_question_meta, rule_b_np, softmax)
from p02_rank_pseudo import (P02Error, is_inside, load_npz, now_iso, sha256_file, str_array,  # noqa: E402
                             write_json)

# ================================================================ constants (PREREG_04 rev 1)
RESULTS = os.path.join(REPO, "results")
P02_RANK = p3.P02_RANK
P03_ROOT = p3.P03_ROOT
P03_OUT = p3.P03_OUT
P04_ROOT = os.path.join(RESULTS, "prereg04")
P04_OUT = os.path.join(P04_ROOT, "rank")
OPP_DIR = os.path.join(P04_ROOT, "opp")
DESIGN_DIR = os.path.join(P04_ROOT, "design")
SPLIT_CSV = p3.SPLIT_CSV
QUESTIONS = p3.QUESTIONS
DATA_P04 = os.path.join(REPO, "data", "prereg04")
COUNTS_NPZ = os.path.join(DATA_P04, "counts_fit.npz")                      # p04_opp.py count (gitignored)
DESIGN_SCRATCH = os.path.join(DATA_P04, "design_scratch")
DESIGN_PROXY = os.path.join(DESIGN_SCRATCH, "proxy_inner_all.npz")
DESIGN_COUNTS = os.path.join(DESIGN_SCRATCH, "counts_fit.npz")
SCRATCH_MANIFEST = os.path.join(DESIGN_DIR, "scratch_manifest.json")
READONLY = [os.path.join(RESULTS, d) for d in ("prereg01", "prereg02", "prereg03", "test02")] + \
    [DESIGN_DIR, OPP_DIR, os.path.join(REPO, "data"), os.path.join(REPO, "docs"), os.path.join(REPO, "colab"),
     os.path.join(REPO, "tools")]

# p04_opp.py outputs read here (registration 1.2; plan.md 5): the interface between the two tools
#   inner_o4.npz / val_o4.npz: key arrays in prereg02's inner_keys / val_keys order, 'o4' and 't4' [N,10] float64,
#     'scored' (inner) / 'replay_failed' (val); inner_report.json / val_report.json: 'status', 'preview',
#     'output_sha256', 'tables_sha256', 'tool'.files_sha256 (val also 'frozen_config'.sha256); count_report.json:
#     games, target_moves, opponent_moves, anomalies, n_cells, counts_sha256, status, preview; dupcheck.json:
#     validation_vs_fit {slots, slots_in_fit_any, slots_in_fit_same_target_colour, slots_in_fit_other_target_colour,
#     share_of_slots, stop}, status; selftest.json: format, passed, problems, preview
O4_INNER, O4_INNER_REPORT = "inner_o4.npz", "inner_report.json"
O4_VAL, O4_VAL_REPORT = "val_o4.npz", "val_report.json"
COUNT_REPORT = "count_report.json"
DUPCHECK_REPORT = "dupcheck.json"
SELFTEST_REPORT = "selftest.json"
SELFTEST_FORMAT = "c10-prereg04 opp selftest v1"                # p04_opp.FORMAT["selftest"]
O4_VARIANT, T4_VARIANT = "o_choice_a20", "t_choice_a20"        # design score_inner layout names
SIDE_ARRAYS = (("o4", "t4"),)                                  # p04_opp.py's layout

INNER_GAMES = 73556
VAL_GAMES = p3.VAL_GAMES_FULL
R20_INNER_OOF = p3.R20_INNER_OOF                                # 0.5854702521055817
DESIGN_OOF = {"Q01": 0.6042484782462081, "Q02": 0.6101211982700248}
SELFCHECK_TOL = 1e-12
MONO_SHA256 = "45d0f14acd98786adfd7cae040edc551420f095aaf6118d52edb47ddf91a5622"
MONO_QUESTIONS = 120544
O4_PINNED = {"games": 720307, "target_moves": 79985811, "opponent_moves": 79985352, "anomalies": 0, "n_cells": 557928}
DESIGN_PROXY_SHA256 = "3e6e3efd90816fb8770691fd1786840e834a66cbccebcd4c2afa9da8ce163b34"
DESIGN_COUNTS_SHA256 = "f6c6debe90639585763b9812c235743a6b7f89d8db1c6422ece7bccd550a08a5"
SW_SHA256 = {   # PREREG_04 1.3
    "a": {"inner": "743825209569aeb922b0246587ec5801f3c2e0a10083bfb6111a327dd5ff1071",
          "val": "b355c26bf68069feb0e48ee95c46bd211c2bdbc6b5a48416c8408a210aec3848"},
    "b": {"inner": "106887321cc9d54316e0d7d96e776c2060dc309c54720a8c7345f280056d2056",
          "val": "0cadff4bae36b2a41f02fddf4366a2081d9912ede4a3063c8b50fd28a2647527"}}
SW_WEIGHTS = {"a": "790b767c5425e262130d6c103c71e89238edb17f927fa48e2a2770db788e50ac",
              "b": "8e88f6ec6cbaad5372e5d62607b99ece3d77dc7a20af86a9197553f1a6c59a42"}
DUP_STOP_FRACTION = 0.01

L = N_LEVELS
B = basis()
ARMS = ("a", "b")
ARM_OF = {"SWA": "a", "SWB": "b", "A0": "a", "B0": "b", "C6A": "a"}
PAIRS = (("O4", "T4"), ("SWA", "A0"), ("SWB", "B0"))      # (opponent-side source, target-side reference)
TERMS = ("OP", "O4", "SWA", "SWB")                         # order of the E terms inside the calibration
COMBOS = (("Q00", ("C5", "OP")), ("Q01", ("C5", "OP", "O4")), ("Q02", ("C5", "OP", "O4", "SWA", "SWB")))
# record only (PREREG_04 2.4): (name, base per-game source, E terms, needed sources, path)
RECORD_ONLY = (("O4 alone", "O4", (), ("O4",), "gen"),
               ("R20+SWA+SWB", "C5", ("OP", "SWA", "SWB"), ("C5", "OP", "SWA", "SWB"), "gen"),
               ("Q02+C6A", "C5", ("OP", "O4", "SWA", "SWB", "C6A"), ("C5", "OP", "O4", "SWA", "SWB", "C6A"), "gen"),
               ("Q00 no opening (R10)", "C5", (), ("C5",), "p02"),
               ("Q01 no opening", "C5", ("O4",), ("C5", "O4"), "gen"),
               ("Q02 no opening", "C5", ("O4", "SWA", "SWB"), ("C5", "O4", "SWA", "SWB"), "gen"))
SOURCE_TEXT = {
    "C5": "prereg02 epoch 9, 8 symmetries (committed; R20's source)",
    "OP": "opening term, prereg02 opening_{inner,val}.npz (M = 12, alpha = 20, fit counts only)",
    "O4": "opponent's moves scored with the target level's pattern-count tables (p04_opp.py, fit counts)",
    "T4": "target's own moves, same tables (colour reference only, never selectable)",
    "SWA": "prereg03 arm A final.pth, original orientation, whole question colour swapped (committed dump)",
    "SWB": "prereg03 arm B final.pth, original orientation, whole question colour swapped (committed dump)",
    "A0": "prereg03 arm A final.pth, symmetry 0, correct colour (colour reference only)",
    "B0": "prereg03 arm B final.pth, symmetry 0, correct colour (colour reference only)",
    "C6A": "prereg03 arm A final.pth, 8 symmetries (record-only combination Q02+C6A)"}
FAIL_OPENING = {"policy": "04 does not play; test02's rule applies (T1 = FAIL_OPENING -> R10)",
                "note": "PREREG_04 2.5: no no-opening fallback of 04 is fitted for use; the no-opening versions are "
                        "recorded only"}
CHECK_FORMAT = "c10-prereg04 rank check_report v1"
SELFCHECK_FORMAT = "c10-prereg04 rank selfcheck v1"
TUNE_FORMAT = "c10-prereg04 rank tune_report v1"
FROZEN_FORMAT = "c10-prereg04 rank frozen_config v1"
APPLY_FORMAT = "c10-prereg04 rank apply_report v1"
OWN_FILE = "p04_rank_fuse.py"
OPP_TOOL = "p04_opp.py"
FIXED_FROZEN = ("p03_rank_fuse.py", "c10data.py", "decide.py")


class P04Error(RuntimeError):
    pass


ERRORS = (P04Error, p3.P03Error, P02Error, decide.DecideError)


# ================================================================ generalised calibration (design fuse.py, verbatim)
def _unpack(th, J):
    return th[0], th[1], th[2:2 + J], th[2 + J:]


def gen_logits(th, m, Es, n):
    J = len(Es)
    a, g, w, beta = _unpack(th, J)
    t = math.exp(a) * np.power(n / 12.0, g)
    u = m + B @ beta
    for j in range(J):
        u = u + w[j] * Es[j]
    return t[:, None] * u, t


def gen_obj(th, m, Es, n, y, wq):
    J = len(Es)
    z, t = gen_logits(th, m, Es, n)
    ls = rf.log_softmax(z)
    Q = len(y)
    a, g, w, beta = _unpack(th, J)
    loss = -np.sum(wq * ls[np.arange(Q), y]) + rf.PENALTY * float(beta @ beta)
    G = np.exp(ls)
    G[np.arange(Q), y] -= 1.0
    G *= wq[:, None]
    gz = np.sum(G * z, axis=1)
    grad = [np.sum(gz), np.sum(gz * np.log(n / 12.0))]
    for j in range(J):
        grad.append(np.sum(t * np.sum(G * Es[j], axis=1)))
    gb = (t[:, None] * G).sum(0)
    return loss, np.concatenate([grad, B.T @ gb + 2 * rf.PENALTY * beta])


def gen_fit_record(m, Es, n, y):
    """gen_fit with the optimiser's record kept (the same computation, in the same order)."""
    J = len(Es)
    wq = rf.level_weights(y)
    bnds = [rf.BOUNDS["a"], rf.BOUNDS["gamma"]] + [rf.BOUNDS["w"]] * J + [(None, None)] * (L - 1)
    best = None
    runs = []
    for a0 in rf.STARTS:
        x0 = np.zeros(2 + J + L - 1)
        x0[0] = a0
        r = optimize.minimize(gen_obj, x0, args=(m, Es, n, y, wq), jac=True, method="L-BFGS-B",
                              bounds=bnds, options={"maxiter": rf.MAXITER})
        runs.append({"start_a": a0, "loss": float(r.fun), "nit": int(r.nit), "success": bool(r.success),
                     "message": str(r.message)})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, {"loss": float(best.fun), "starts": runs}


def gen_fit(m, Es, n, y):
    return gen_fit_record(m, Es, n, y)[0]


def gen_oof(Xf, Xe, mf, me, Ef, Ee):
    P = np.zeros((len(Xe), L))
    ws = []
    for f in range(5):
        ev = Xe.fold == f
        tr = Xf.fold != f
        th = gen_fit(mf[tr], [E[tr] for E in Ef], Xf.n[tr], Xf.y[tr])
        z, _ = gen_logits(th, me[ev], [E[ev] for E in Ee], Xe.n[ev])
        P[ev] = rf.softmax(z)
        ws.append([float(x) for x in th[2:2 + len(Ef)]])
    return P, ws


def gen_oof_record(Xf, Xe, mf, me, Ef, Ee):
    """gen_oof with every fold's parameters and optimiser record kept (same fits, same order)."""
    P = np.zeros((len(Xe), L))
    folds = []
    for f in range(5):
        ev = Xe.fold == f
        tr = Xf.fold != f
        if not tr.any():
            raise P04Error(f"fold {f}: no fit questions")
        th, rec = gen_fit_record(mf[tr], [E[tr] for E in Ef], Xf.n[tr], Xf.y[tr])
        z, _ = gen_logits(th, me[ev], [E[ev] for E in Ee], Xe.n[ev])
        P[ev] = rf.softmax(z)
        rec.update({"fold": f, "fit_questions": int(tr.sum()), "eval_questions": int(ev.sum()),
                    "theta": [float(x) for x in th]})
        folds.append(rec)
    return P, folds


def gen_fold_predict(folds, m, Es, n, fold):
    """Probabilities with each fold's recorded parameters (row q uses the fold of its player)."""
    P = np.full((len(n), L), np.nan)
    for rec in folds:
        ev = fold == rec["fold"]
        if ev.any():
            z, _ = gen_logits(np.asarray(rec["theta"], dtype=np.float64), m[ev], [E[ev] for E in Es], n[ev])
            P[ev] = rf.softmax(z)
    return P


def gen_params(th, terms):
    a, g, w, beta = _unpack(np.asarray(th, dtype=np.float64), len(terms))
    return {"path": "gen", "terms": list(terms), "a": float(a), "gamma": float(g),
            "w": {t: float(x) for t, x in zip(terms, w)}, "beta": [float(x) for x in beta],
            "b": [float(x) for x in B @ beta], "theta": [float(x) for x in th]}


def p02_params(theta):
    """Q00 (prereg02's opening fit) in the same record layout; theta = [a, gamma, w, beta] is the generalised layout
    with E = [OP]."""
    par = params_dict(theta, True, B)
    return {"path": "p02", "terms": ["OP"], "a": par["a"], "gamma": par["gamma"], "w": {"OP": par["w"]},
            "beta": par["beta"], "b": par["b"], "theta": [float(x) for x in theta]}


def as_p02_dict(par):
    return {"a": par["a"], "gamma": par["gamma"], "w": par["w"]["OP"], "beta": par["beta"], "b": par["b"]}


# ================================================================ helpers
def read_json(path):
    return p3.read_json(path)


def rel(path):
    return p3.repo_rel(path)


def frozen_files():
    """tools/p02_*.py (every one in the working tree), p03_rank_fuse.py, c10data.py, decide.py (PREREG_04 2.1)."""
    p02 = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "p02_*.py")))
    return tuple(p02) + FIXED_FROZEN


def frozen_tools_check(blob=None):
    blob = blob or p3.head_blob
    rec, bad = {}, []
    for f in frozen_files():
        path = os.path.join(HERE, f)
        work = p3.sha256_text(path) if os.path.exists(path) else None
        b = blob("tools/" + f)
        head = p3.sha256_bytes_text(b) if b is not None else None
        rec[f] = work
        if work is None or head is None or work != head:
            bad.append(f"{f} (working {str(work)[:12]}, HEAD {str(head)[:12]})")
    if bad:
        raise P04Error(f"frozen tools differ from HEAD: {bad}; the p02 / p03 tools, c10data.py and decide.py are never "
                       f"edited (PREREG_04 2.1)")
    return {"rule": "sha256 with CRLF folded to LF, working file == HEAD blob", "files_sha256": rec}


def tool_state():
    files = [OWN_FILE] + ([OPP_TOOL] if os.path.exists(os.path.join(HERE, OPP_TOOL)) else []) + list(frozen_files())
    rels = ["tools/" + f for f in files]
    rc, st = p3.git(["status", "--porcelain", "--untracked-files=all", "--"] + rels, binary=True)
    lines = st.decode("utf-8", "replace").splitlines() if rc == 0 else None
    return {"git_head": p3.git_head(),
            "dirty_or_untracked": [ln[3:] for ln in lines if ln.strip()] if lines is not None else None,
            "files_sha256_rule": "sha256 of the file with CRLF folded to LF",
            "files_sha256": {f: p3.sha256_text(os.path.join(HERE, f)) for f in files}}


def require_tools_committed(tools):
    if tools["git_head"] is None or tools["dirty_or_untracked"] is None or tools["dirty_or_untracked"]:
        raise P04Error(f"commit the tools first (HEAD {tools['git_head']}, uncommitted {tools['dirty_or_untracked']})")


def environment():
    return p3.environment()


def ci95(d):
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ================================================================ paths and modes
class Ctx:
    pass


def resolve(a, cmd):
    """Real mode: repository inputs; out-dir results/prereg04/rank, or outside the repository for a preview of
    check / selfcheck / tune. --smoke: every input path given, output outside the repository. --design-o4: preview
    only, O4 / T4 inner vectors from the design file."""
    c = Ctx()
    c.cmd = cmd
    c.smoke = bool(a.smoke)
    c.design = bool(getattr(a, "design_o4", False))
    if c.smoke and c.design:
        raise P04Error("--design-o4 reads the real inner data; it cannot be combined with --smoke")
    given = {"--p02-dir": a.p02_dir, "--p02-local-dir": a.p02_local_dir, "--p03-dir": a.p03_dir,
             "--opp-dir": a.opp_dir, "--split": a.split, "--questions": a.questions}
    real = {"--p02-dir": P02_RANK, "--p02-local-dir": P02_RANK, "--p03-dir": P03_ROOT, "--opp-dir": OPP_DIR,
            "--split": SPLIT_CSV, "--questions": QUESTIONS}
    if c.smoke:
        need = ["--p02-dir", "--p02-local-dir", "--p03-dir", "--opp-dir", "--split"] + \
            (["--questions"] if cmd == "apply" else [])
        missing = [k for k in need if not given[k]] + ([] if a.out_dir else ["--out-dir"])
        if missing:
            raise P04Error(f"--smoke needs {missing} (smoke never falls back to the repository paths)")
        vals = {k: (os.path.abspath(v) if v else None) for k, v in given.items()}
        c.out_dir = os.path.abspath(a.out_dir)
        if is_inside(c.out_dir, REPO):
            raise P04Error(f"--smoke output must be outside the repository (got {c.out_dir})")
        for k in ("--p02-dir", "--p02-local-dir", "--p03-dir", "--opp-dir"):
            if is_inside(vals[k], REPO):
                raise P04Error(f"--smoke {k} must be outside the repository (synthetic data only; got {vals[k]})")
    else:
        for k, v in given.items():
            if v and not p3.same_path(v, real[k]):
                raise P04Error(f"{k}: real mode uses {real[k]} only (use --smoke for synthetic data)")
        vals = dict(real)
        c.out_dir = os.path.abspath(a.out_dir) if a.out_dir else P04_OUT
        if is_inside(c.out_dir, REPO) and not p3.same_path(c.out_dir, P04_OUT):
            raise P04Error(f"--out-dir inside the repository must be {P04_OUT} (got {c.out_dir})")
        if cmd == "apply" and not p3.same_path(c.out_dir, P04_OUT):
            raise P04Error(f"apply writes {P04_OUT} only (no preview; smoke runs use --smoke)")
    c.in_repo = is_inside(c.out_dir, REPO)
    c.preview = (not c.smoke) and not c.in_repo
    if c.design and not c.preview:
        raise P04Error("--design-o4 is a preview: give an --out-dir outside the repository (and never apply)")
    for ro in READONLY:
        if is_inside(c.out_dir, ro):
            raise P04Error(f"refusing to write into {c.out_dir}: {ro} is read-only for prereg04's fusion tool")
    c.p02_dir, c.p02_local, c.p03_dir = vals["--p02-dir"], vals["--p02-local-dir"], vals["--p03-dir"]
    c.opp_dir, c.split, c.questions = vals["--opp-dir"], vals["--split"], vals["--questions"]
    c.p02_dumps = os.path.join(c.p02_dir, "dumps")
    c.arm_dir = {x: os.path.join(c.p03_dir, f"rank_{x}") for x in ARMS}
    for d in (c.p02_dir, c.p02_local, c.p03_dir, c.opp_dir) + tuple(c.arm_dir.values()):
        if is_inside(c.out_dir, d):
            raise P04Error(f"refusing to write into the input folder {d}")
    flag = p3.notebook_smoke_flag(c.p02_dir)
    if c.smoke and flag is not True:
        raise P04Error(f"--smoke needs synthetic / SMOKE prereg02 dumps ({c.p02_dir}/config.json smoke {flag!r})")
    if not c.smoke and flag is not False:
        raise P04Error(f"{c.p02_dir}/config.json must say \"smoke\": false in real mode (got {flag!r})")
    return c


def mode_flags(c):
    return {"smoke": c.smoke, "preview": c.preview, "design_input": c.design}


def start(c):
    """Common prologue: imports from tools/, frozen tool hashes (HEAD; real mode also prereg02's and prereg03's
    records)."""
    p3.imported_from_tools()
    if not p3.same_path(p3.__file__, os.path.join(HERE, "p03_rank_fuse.py")):
        raise P04Error(f"p03_rank_fuse imported from {p3.__file__}, not {HERE}")
    ft = frozen_tools_check()
    if not c.smoke:
        for what, path in (("prereg02", os.path.join(P02_RANK, "frozen_config.json")),
                           ("prereg03", os.path.join(P03_OUT, "frozen_config.json"))):
            rec = (read_json(path).get("tool") or {}).get("files_sha256") or {}
            diff = [f for f, h in rec.items() if f in ft["files_sha256"] and ft["files_sha256"][f] != h]
            missing = [f for f in rec if f not in ft["files_sha256"]]
            if diff or missing:
                raise P04Error(f"frozen tools differ from the hashes {what}'s tune recorded: {diff + missing}")
            ft[f"equal_to_{what}_frozen_config_record"] = sorted(rec)
    return ft


def committed_inputs(c, val=False):
    """Real mode (results/prereg04/rank): p04_opp.py's committed inner outputs and the prereg03 dumps used here are
    tracked by git and unchanged (the validation O4 is committed only after apply, PREREG_04 9 step 4)."""
    if c.smoke or c.preview:
        return None
    paths = [os.path.join(c.opp_dir, f) for f in (O4_INNER, O4_INNER_REPORT, COUNT_REPORT, DUPCHECK_REPORT,
                                                  SELFTEST_REPORT)]
    for x in ARMS:
        d = os.path.join(c.arm_dir[x], "dumps")
        paths += [os.path.join(d, f"{s}.npz") for s in ("inner_keys", "val_keys", f"inner_p03{x}_d4",
                                                          f"inner_p03{x}_s0_swap")]
        if val:
            paths += [os.path.join(d, f"{s}.npz") for s in (f"val_p03{x}_d4", f"val_p03{x}_s0_swap")]
    return p3.require_tracked_clean(paths, "p04_opp.py outputs and prereg03 dumps (commit count / inner / dupcheck "
                                           "before check)")


def val_o4_order_guard(c):
    """PREREG_04 1.2: the validation O4 may exist only after frozen_config.json is committed (tracked by git and
    unchanged in the working tree, as p04_opp.require_frozen_config and p3.require_tracked_clean)."""
    if c.smoke:
        return None
    vpath = os.path.join(c.opp_dir, O4_VAL)
    if not os.path.exists(vpath):
        return None
    fpath = os.path.join(P04_OUT, "frozen_config.json")
    r = rel(fpath)
    rc, _ = p3.git(["ls-files", "--error-unmatch", "--", r])
    why = None
    if rc != 0:
        why = "not committed"
    else:
        rc, st = p3.git(["status", "--porcelain", "--", r])
        if rc != 0:
            why = "not verifiable (git status failed)"
        elif st:
            why = f"changed in the working tree or only staged ({st[:80]})"
    if why:
        shown = rel(vpath) if is_inside(vpath, REPO) else vpath
        return (f"{shown} exists but results/prereg04/rank/frozen_config.json is {why}: the validation O4 is "
                f"computed only after the frozen config is committed (PREREG_04 1.2); stop and find out why")
    return None


def selftest_status(c):
    """PREREG_04 9 / plan.md 5: p04_opp.py selftest passed (0 capture and 0 feature mismatches) before its outputs are
    used. Real mode only (smoke worlds have no selftest; --design-o4 previews run before p04_opp.py)."""
    path = os.path.join(c.opp_dir, SELFTEST_REPORT)
    if not os.path.exists(path):
        return [f"selftest: {path} missing (run p04_opp.py selftest and commit selftest.json with count)"], \
            {"file": path, "present": False}
    rep = read_json(path)
    cap = rep.get("capture") or {}
    fea = rep.get("features") or {}
    rec = {"file": path, "present": True, "sha256": sha256_file(path), "format": rep.get("format"),
           "passed": rep.get("passed"), "preview": rep.get("preview"), "problems": rep.get("problems"),
           "capture_games_with_mismatch": cap.get("games_with_mismatch"), "feature_positions": fea.get("positions"),
           "feature_mismatches": fea.get("mismatches")}
    probs = []
    if rep.get("format") != SELFTEST_FORMAT:
        probs.append(f"selftest: format {rep.get('format')!r}, expected {SELFTEST_FORMAT!r}")
    if rep.get("passed") is not True or rep.get("problems") != []:
        probs.append(f"selftest: passed {rep.get('passed')!r}, problems {rep.get('problems')!r} (must pass with none)")
    if rep.get("preview") is not False:
        probs.append(f"selftest: preview {rep.get('preview')!r}; real mode needs the registered run")
    if cap.get("games_with_mismatch") != 0 or fea.get("mismatches") != 0:
        probs.append(f"selftest: capture mismatches {cap.get('games_with_mismatch')!r}, feature mismatches "
                     f"{fea.get('mismatches')!r} (registered 0 / 0)")
    return probs, rec


# ================================================================ O4 / T4 files (p04_opp.py outputs)
def side_vectors(z, path):
    """O4 and T4 per-game 10-vectors [N,10] float64 (sum over that side's moves of log P_k): p04_opp.py's 'o4' / 't4'
    arrays, or (design file) the score_inner layout 'll' [N,V,3,10] with 'variants' -> ll[:, v].sum(1) for
    v = o_choice_a20 / t_choice_a20, exactly as the design scripts and p04_opp.side_vectors."""
    for on, tn in SIDE_ARRAYS:
        if on in z and tn in z:
            o, t = np.asarray(z[on]), np.asarray(z[tn])
            layout = f"{on}/{tn}"
            break
    else:
        if "ll" in z and "variants" in z:
            V = str_array(z["variants"])
            if O4_VARIANT not in V or T4_VARIANT not in V:
                raise P04Error(f"{path}: variants {V} lack {O4_VARIANT} / {T4_VARIANT}")
            ll = np.asarray(z["ll"])
            if ll.ndim != 4 or ll.shape[2:] != (3, L):
                raise P04Error(f"{path}: ll shape {ll.shape}, expected [N, V, 3, 10]")
            o, t = ll[:, V.index(O4_VARIANT)], ll[:, V.index(T4_VARIANT)]
            layout = "design ll[:, variant] summed over the 3 phases"
        else:
            raise P04Error(f"{path}: no O4 / T4 arrays (expected one of {SIDE_ARRAYS} or 'll' + 'variants')")
    out = []
    for name, v in (("O4", o), ("T4", t)):
        if v.dtype != np.float64:
            raise P04Error(f"{path}: {name} must be float64, got {v.dtype}")
        if v.ndim == 3 and v.shape[1:] == (3, L):
            v = v.sum(1)
        if v.ndim != 2 or v.shape[1] != L:
            raise P04Error(f"{path}: {name} shape {v.shape}, expected [N, 10]")
        out.append(v)
    return out[0], out[1], layout


def load_o4_inner(c, keys):
    """-> (problems, record, {'O4': [N,10], 'T4': [N,10]} or None) for the inner games in inner_keys order."""
    probs = []
    if c.design:
        path = DESIGN_PROXY
        rec = {"file": path, "design_input": True,
               "note": "DESIGN INPUT (preview only): data/prereg04/design_scratch/proxy_inner_all.npz stands in for "
                       "p04_opp.py's registered inner output"}
    else:
        path = os.path.join(c.opp_dir, O4_INNER)
        rec = {"file": path}
    if not os.path.exists(path):
        return [f"O4: {path} missing (run p04_opp.py inner first)"], rec, None
    z = load_npz(path, [])
    rec["sha256"] = sha256_file(path)
    if c.design:
        if rec["sha256"] != DESIGN_PROXY_SHA256:
            probs.append(f"O4 design file SHA-256 {rec['sha256']} != pinned {DESIGN_PROXY_SHA256}")
        man = read_json(SCRATCH_MANIFEST)
        entry = next((e for e in man.get("large_files", []) if e.get("file", "").endswith("/proxy_inner_all.npz")), {})
        rec["scratch_manifest_sha256"] = entry.get("sha256")
        if entry.get("sha256") != DESIGN_PROXY_SHA256:
            probs.append("O4 design file: scratch_manifest.json records another SHA-256")
    o, t, layout = side_vectors(z, path)
    rec["layout"] = layout
    names = ("rank", "game_id", "player_id", "color")
    miss = [k for k in names if k not in z]
    if miss:
        return probs + [f"O4: {path} lacks key arrays {miss}"], rec, None
    got = {k: str_array(z[k]) for k in names}
    n = keys["n"]
    if c.design:
        row = {(r, g): i for i, (r, g) in enumerate(zip(got["rank"], got["game_id"]))}
        idx = [row.get((r, g)) for r, g in zip(keys["rank"], keys["game_id"])]
        if any(i is None for i in idx) or len(got["rank"]) != n:
            return probs + ["O4 design file does not cover inner_keys"], rec, None
        idx = np.asarray(idx)
        bad = [k for k in ("player_id", "color") if [got[k][i] for i in idx] != keys[k]]
        if bad:
            probs.append(f"O4 design file: {bad} differ from inner_keys after aligning (rank, game_id)")
        o, t = o[idx], t[idx]
        rec["key_alignment"] = "design file re-indexed by (rank, game_id) as the design scripts did"
        if "scored" in z and not np.asarray(z["scored"]).all():
            probs.append("O4 design file: not every inner game scored")
    else:
        diff = [k for k in names if got[k] != keys[k]]
        rec["key_arrays_equal_inner_keys"] = not diff
        if diff:
            probs.append(f"O4: key arrays {diff} differ from prereg02's inner_keys.npz (PREREG_04 2.2 item 1)")
        if "scored" not in z or not np.asarray(z["scored"]).all():
            probs.append("O4: not every inner game was scored (a subset preview or replay failures)")
        if "anomalies" in z and int(np.asarray(z["anomalies"]).sum()):
            probs.append("O4: inner anomalies (registered 0)")
        p, rec["report"] = opp_report(c, O4_INNER_REPORT, rec["sha256"])
        probs.extend(p)
    rec["rows"] = int(len(o))
    if len(o) != n or len(t) != n:
        probs.append(f"O4: {len(o)} rows, inner_keys has {n}")
        return probs, rec, None
    if not c.smoke and n != INNER_GAMES:
        probs.append(f"O4: {n} inner games, expected {INNER_GAMES}")
    fin = bool(np.isfinite(o).all() and np.isfinite(t).all())
    rec["finite"] = fin
    if not fin:
        probs.append("O4 / T4: non-finite inner values (the NaN pattern must equal C5's: none on inner)")
    return probs, rec, {"O4": o, "T4": t}


def opp_report(c, name, output_sha):
    """A p04_opp.py report next to its output: status ok, a real run in real mode, output_sha256 = the file read."""
    path = os.path.join(c.opp_dir, name)
    if not os.path.exists(path):
        return [f"{name}: missing next to the p04_opp.py output"], {"file": path, "present": False}
    rep = read_json(path)
    rec = {"file": path, "sha256": sha256_file(path), "status": rep.get("status"), "preview": rep.get("preview"),
           "output_sha256": rep.get("output_sha256"), "tables_sha256": rep.get("tables_sha256"),
           "p04_opp_sha256": ((rep.get("tool") or {}).get("files_sha256") or {}).get(OPP_TOOL),
           "frozen_config_sha256": (rep.get("frozen_config") or {}).get("sha256")
           if isinstance(rep.get("frozen_config"), dict) else None}
    probs = []
    if rep.get("status") != "ok":
        probs.append(f"{name}: status {rep.get('status')!r} (p04_opp.py stopped)")
    if not c.smoke and rep.get("preview") is not False:
        probs.append(f"{name}: preview {rep.get('preview')!r}; real mode needs the registered run")
    if output_sha is not None and rep.get("output_sha256") != output_sha:
        probs.append(f"{name}: output_sha256 differs from the file read")
    if not c.smoke and not is_sha256(rep.get("tables_sha256")):
        probs.append(f"{name}: no tables_sha256 (the rate tables' SHA-256 is frozen by tune, compared by apply)")
    return probs, rec


def is_sha256(x):
    return isinstance(x, str) and len(x) == 64 and all(ch in "0123456789abcdef" for ch in x)


def load_o4_val(c, vk):
    """-> (problems, record, {'O4', 'T4'} [n_slots,10] (0 on unparsed rows) or None) for the validation slots.
    Required: replay_failed false on every slot and finite O4 / T4 on every slot C5 parsed. Slots C5 did not parse are
    ignored (set to 0, never used: E_j pools the parsed games only); they need not be NaN (p04_opp.py val scores every
    slot it can replay). This replaces 'NaN pattern equal to C5's' of PREREG_04 2.2 item 1 for the validation O4
    (PREREG_04 rev 2)."""
    path = os.path.join(c.opp_dir, O4_VAL)
    rec = {"file": path}
    if not os.path.exists(path):
        return [f"O4: {path} missing (run p04_opp.py val after committing frozen_config.json)"], rec, None
    z = load_npz(path, [])
    rec["sha256"] = sha256_file(path)
    o, t, layout = side_vectors(z, path)
    rec["layout"] = layout
    probs, rec["report"] = opp_report(c, O4_VAL_REPORT, rec["sha256"])
    miss = [k for k in ("question_id", "slot", "color") if k not in z]
    if miss:
        return probs + [f"O4 val: {path} lacks key arrays {miss}"], rec, None
    if str_array(z["question_id"]) != vk["question_id"] or \
            [int(x) for x in np.asarray(z["slot"]).tolist()] != vk["slot"] or str_array(z["color"]) != vk["color"]:
        probs.append("O4 val: question_id / slot / color differ from prereg02's val_keys.npz")
    if len(o) != vk["n"]:
        return probs + [f"O4 val: {len(o)} rows, val_keys has {vk['n']}"], rec, None
    parsed = vk["parsed"]
    fin = np.isfinite(o).all(1) & np.isfinite(t).all(1)
    failed = np.asarray(z["replay_failed"]).astype(bool) if "replay_failed" in z else ~fin
    rec["replay_failures"] = int(failed.sum())
    rec["non_finite_on_parsed_slots"] = int((~fin[parsed]).sum())
    rec["unparsed_slots_ignored"] = int((~parsed).sum())
    if failed.any() or not fin[parsed].all():
        probs.append(f"O4 val: {int(failed.sum())} replay failures, {int((~fin[parsed]).sum())} parsed slots without a "
                     f"finite O4 (registered 0; every slot C5 parsed must have a finite O4 / T4)")
    rec["rows"] = int(len(o))
    mask = parsed[:, None]
    return probs, rec, {"O4": np.where(mask, o, 0.0), "T4": np.where(mask, t, 0.0)}


def o4_pinned(c, o4_inner):
    """PREREG_04 1.2 pinned numbers (real, non-design mode): count report, counts arrays, inner vs design."""
    probs = []
    rec = {}
    rpath = os.path.join(c.opp_dir, COUNT_REPORT)
    if not os.path.exists(rpath):
        return [f"O4: {rpath} missing (run p04_opp.py count)"], {"count_report": None}
    rep = read_json(rpath)
    facts = {k: rep.get(k) for k in O4_PINNED}
    rec["count_report"] = {"file": rpath, "sha256": sha256_file(rpath), "facts": facts, "status": rep.get("status"),
                           "preview": rep.get("preview"), "counts_sha256": rep.get("counts_sha256")}
    if rep.get("status") != "ok":
        probs.append(f"O4: count_report.json status {rep.get('status')!r}")
    if c.smoke:
        rec["pinned"] = "recorded only (--smoke)"
        return probs, rec
    if rep.get("preview") is not False:
        probs.append(f"O4: count_report.json preview {rep.get('preview')!r}; real mode needs the registered count")
    bad = {k: (facts[k], v) for k, v in O4_PINNED.items() if facts[k] != v}
    if bad:
        probs.append(f"O4 pinned numbers differ (report, pinned): {bad}")
    # counts arrays (gitignored) equal to the design counts, array by array (recomputed here when both large files are
    # on this machine; otherwise count_report.json's own record of that comparison, which must say equal)
    rec_eq = ((rep.get("design_counts") or {}).get("arrays_equal") or {}) if isinstance(rep.get("design_counts"),
                                                                                         dict) else {}
    if os.path.exists(COUNTS_NPZ) and os.path.exists(DESIGN_COUNTS):
        sha = sha256_file(COUNTS_NPZ)
        rec["counts_npz"] = {"file": COUNTS_NPZ, "sha256": sha, "via": "recomputed here",
                             "equal_to_count_report_counts_sha256": sha == rep.get("counts_sha256")}
        if sha != rep.get("counts_sha256"):
            probs.append("O4: data/prereg04/counts_fit.npz SHA-256 differs from count_report.json counts_sha256")
        dsha = sha256_file(DESIGN_COUNTS)
        if dsha != DESIGN_COUNTS_SHA256:
            probs.append(f"O4: design counts SHA-256 {dsha} != pinned {DESIGN_COUNTS_SHA256}")
        with np.load(COUNTS_NPZ, allow_pickle=False) as zc, np.load(DESIGN_COUNTS, allow_pickle=False) as zd:
            names = sorted(zd.files)
            diff = [k for k in names if k not in zc.files or not np.array_equal(zc[k], zd[k])]
            extra = sorted(set(zc.files) - set(zd.files))
            cells = int(zd[names[0]].shape[-1]) if names else None
        rec["counts_npz"].update({"equal_to_design_array_by_array": not diff, "arrays": names,
                                  "differing": diff, "extra_arrays": extra, "cells": cells})
        if diff:
            probs.append(f"O4: counts arrays {diff} differ from the design counts_fit.npz")
    else:
        ok = bool(rec_eq) and all(v is True for v in rec_eq.values()) and \
            set(rec_eq) == {"C_t", "A_t", "C_o", "A_o"}
        rec["counts_npz"] = {"via": "count_report.json design_counts.arrays_equal (the large files are not both on "
                                    "this machine)", "record": rec_eq, "equal": ok,
                             "present": {"counts": os.path.exists(COUNTS_NPZ), "design": os.path.exists(DESIGN_COUNTS)}}
        if not ok:
            probs.append("O4: no evidence that the counts equal the design counts_fit.npz array by array")
    # the inner O4 / T4 vectors within 1e-12 of the design file (recomputed; else inner_report.json's record)
    if o4_inner is not None:
        if not os.path.exists(DESIGN_PROXY):
            rpt = os.path.join(c.opp_dir, O4_INNER_REPORT)
            cmp_ = ((read_json(rpt).get("design_proxy") or {}).get("compare") or {}) if os.path.exists(rpt) else {}
            d = {k: (cmp_.get(v) or {}).get("max_abs_diff") for k, v in (("O4", O4_VARIANT), ("T4", T4_VARIANT))}
            rec["inner_vs_design_max_abs_diff"] = dict(d, via="inner_report.json design_proxy.compare (design file "
                                                              "not on this machine)")
            if any(not isinstance(x, (int, float)) or x > SELFCHECK_TOL for x in d.values()):
                probs.append(f"O4: no evidence that the inner vectors are within 1e-12 of proxy_inner_all.npz ({d})")
        else:
            dc = Ctx()
            dc.design, dc.smoke, dc.opp_dir = True, False, None
            p, drec, dv = load_o4_inner(dc, o4_inner["keys"])
            if p or dv is None:
                probs.extend(f"design comparison: {x}" for x in p)
            else:
                d = {k: float(np.max(np.abs(o4_inner[k] - dv[k]))) for k in ("O4", "T4")}
                rec["inner_vs_design_max_abs_diff"] = d
                if max(d.values()) > SELFCHECK_TOL:
                    probs.append(f"O4: inner vectors differ from proxy_inner_all.npz by {d} (> 1e-12)")
    return probs, rec


def dupcheck_status(c):
    path = os.path.join(c.opp_dir, DUPCHECK_REPORT)
    if not os.path.exists(path):
        return [f"dupcheck: {path} missing (run p04_opp.py dupcheck before check)"], {"file": path, "present": False}
    rep = read_json(path)
    val = rep.get("validation_vs_fit") or {}
    total, hits = val.get("slots"), val.get("slots_in_fit_any")
    rec = {"file": path, "present": True, "sha256": sha256_file(path), "status": rep.get("status"),
           "preview": rep.get("preview"), "slots": total, "slots_in_fit_any": hits,
           "slots_in_fit_same_target_colour": val.get("slots_in_fit_same_target_colour"),
           "slots_in_fit_other_target_colour": val.get("slots_in_fit_other_target_colour"),
           "inner_vs_fit_record_only": rep.get("inner_vs_fit_record_only"), "stop_fraction": DUP_STOP_FRACTION}
    if not isinstance(total, int) or not isinstance(hits, int) or total <= 0:
        return [f"dupcheck: {path} has no validation_vs_fit slots / slots_in_fit_any"], rec
    frac = hits / total
    rec["fraction"] = frac
    probs = []
    if not c.smoke and total != VAL_GAMES:
        probs.append(f"dupcheck: {total} slots, the validation questions have {VAL_GAMES}")
    if not c.smoke and rep.get("preview") is not False:
        probs.append(f"dupcheck: preview {rep.get('preview')!r}; real mode needs the registered run")
    if frac >= DUP_STOP_FRACTION:
        probs.append(f"dupcheck: {hits} of {total} validation slots ({frac:.2%}) also in fit (same or other target "
                     f"colour) >= 1 %: stop, write a revision before tune (PREREG_04 2.2 item 3)")
    if val.get("stop") is not False or rep.get("status") != "ok":
        probs.append(f"dupcheck: report stop {val.get('stop')!r} / status {rep.get('status')!r}")
    if hits:
        rec["disclose"] = f"{hits} validation slots also in fit: record and disclose (PREREG_04 2.2 item 3)"
    return probs, rec


# ================================================================ inner inputs
def inner_setup(c):
    """prereg02's inner pseudo-questions (fit / eval rounds), single-colour pseudo-questions, bootstrap draws."""
    inner = p3.prepare_inner(c)
    ms = p3.mono_set(inner)
    if not c.smoke and (ms["sha256"] != MONO_SHA256 or len(ms["questions"]) != MONO_QUESTIONS):
        raise P04Error(f"single-colour inner pseudo-questions: SHA-256 {ms['sha256']} / {len(ms['questions'])} "
                       f"questions, registered {MONO_SHA256} / {MONO_QUESTIONS} (PREREG_03 rev 3 (a))")
    if not ms["questions"]:
        raise P04Error("no single-colour inner pseudo-question")
    Xm, Sm = p3.mono_qset(inner, ms)
    inner.update(ms=ms, X_mono=Xm, S_mono=Sm, mono_boot=PlayerBootstrap(ms["player"], ms["y"], ms["n_players"]))
    return inner


def arm_inner(c, x, n):
    ddir = os.path.join(c.arm_dir[x], "dumps")
    d4 = load_dump(os.path.join(ddir, f"inner_p03{x}_d4.npz"), n, p3.N_SYMS)
    sw = p3.load_swap(os.path.join(ddir, f"inner_p03{x}_s0_swap.npz"), n)
    return d4, sw


def inner_sources(c, keys, need, meta=None):
    """Per-game log-probabilities of the inner games for the sources in need (+ SHA-256 records). meta (a dict), when
    given, receives the O4 file record and the raw O4 / T4 vectors."""
    n = keys["n"]
    lp, sha, probs = {}, {}, []
    meta = {} if meta is None else meta
    c5 = p3.load_c5_inner(c, n)
    lp["C5"] = p3.combo_logprobs(("C5",), {p3.C5_DUMP: c5["logits"]})
    sha[f"inner_{p3.C5_DUMP}.npz"] = c5["sha256"]
    meta["C5_weights_sha256"] = c5["weights_sha256"]
    if {"O4", "T4"} & set(need):
        p, orec, ov = load_o4_inner(c, keys)
        probs.extend(p)
        sha["O4_inner"] = orec.get("sha256")
        meta["O4_file"] = orec
        if ov is not None:
            lp["O4"], lp["T4"] = log_softmax(ov["O4"]), log_softmax(ov["T4"])
            meta["O4_raw"] = ov
    for x in ARMS:
        want = [s for s in ("SWA", "SWB", "A0", "B0", "C6A") if ARM_OF[s] == x and s in need]
        if not want:
            continue
        d4, sw = arm_inner(c, x, n)
        sha[f"inner_p03{x}_d4.npz"] = d4["sha256"]
        sha[f"inner_p03{x}_s0_swap.npz"] = sw["sha256"]
        S = x.upper()
        if f"SW{S}" in want:
            lp[f"SW{S}"] = log_softmax(sw["logits"].astype(np.float64))
        if f"{S}0" in want:
            lp[f"{S}0"] = log_softmax(d4["logits"][:, 0, :].astype(np.float64))
        if "C6A" in want and x == "a":
            lp["C6A"] = p3.combo_logprobs(("C6A",), {"p03a_d4": d4["logits"]})
    return lp, sha, probs


def question_terms(X, S, lp, terms):
    """E arrays of a question set in the order of terms (OP = the set's opening mean)."""
    return [X.mop if t == "OP" else pooled(S, lp[t], "log") for t in terms]


# ================================================================ colour semantics (PREREG_04 2.2 item 2)
def side_kappa(lp_ref, lp_src, inner, vk=None, vlp_ref=None, vlp_src=None):
    """dkappa = kappa(target-side reference) - kappa(opponent-side source) on the single-colour inner pseudo-questions
    (PlayerBootstrap CI of the plain question mean) and, when given, on the validation single-colour sub-questions."""
    ms, boot = inner["ms"], inner["mono_boot"]
    kr, ks = p3.kappa_questions(lp_ref, ms["groups"]), p3.kappa_questions(lp_src, ms["groups"])
    dk = kr - ks
    if np.isnan(dk).any():
        raise P04Error("single-colour questions with < 2 games (every one has >= 5)")
    rec = {"inner": {"reference": float(np.mean(kr)), "source": float(np.mean(ks)), "dkappa": float(np.mean(dk)),
                     "ci95": ci95(p3.boot_plain_mean(boot, dk)), "questions": int(len(dk)),
                     "questions_sha256": ms["sha256"],
                     "bootstrap": f"PlayerBootstrap draws (players within each level, {N_BOOT}, "
                                  f"default_rng({BOOT_SEED})) of the plain question mean; colour splits draw their own",
                     "by_colour": {}, "by_k": {}}}
    for col in p3.MONO_COLOURS:
        mk = ms["colour"] == col
        rec["inner"]["by_colour"][col] = ({"dkappa": float(np.mean(dk[mk])), "questions": int(mk.sum()),
                                           "ci95": ci95(p3.boot_plain_mean(p3.sub_bootstrap(ms, mk), dk[mk]))}
                                          if mk.any() else None)
    for k in p3.MONO_KS:
        mk = ms["k"] == k
        rec["inner"]["by_k"][str(k)] = {"dkappa": float(np.mean(dk[mk])), "questions": int(mk.sum())} if mk.any() \
            else None
    rec["val_sub_questions"] = val_side_kappa(vk, vlp_ref, vlp_src) if vk is not None and vlp_ref is not None \
        else None
    return rec


def val_side_kappa(vk, vlp_ref, vlp_src):
    sub = p3.val_colour_subquestions(vk)
    if not sub:
        return {"sub_questions": 0, "dkappa": None}
    groups = [g for _, _, g in sub]
    qidx = np.array([q for q, _, _ in sub])
    cols = np.array([col for _, col, _ in sub])
    kr, ks = p3.kappa_questions(vlp_ref, groups), p3.kappa_questions(vlp_src, groups)
    dk = kr - ks
    out = {"reference": float(np.mean(kr)), "source": float(np.mean(ks)), "dkappa": float(np.mean(dk)),
           "sub_questions": int(len(sub)), "questions_with_a_sub_question": int(len(np.unique(qidx))),
           "question_bootstrap_ci95_recorded_only": ci95(p3.question_boot_mean(qidx, dk)),
           "definition": "each validation question's parsed games split by the question file's colour (val_keys.npz; "
                         "no label); sub-questions with >= 2 parsed games; plain mean over sub-questions",
           "by_colour": {}}
    for col in p3.MONO_COLOURS:
        mk = cols == col
        out["by_colour"][col] = {"dkappa": float(np.mean(dk[mk])), "sub_questions": int(mk.sum())} if mk.any() else None
    return out


def colour_usable(src, rec):
    lo = rec["inner"]["ci95"][0]
    reasons = []
    if not lo > 0:
        reasons.append(f"{src}: single-colour inner dkappa CI lower bound {lo:.4g} not > 0")
    if src != "O4":
        v = (rec.get("val_sub_questions") or {}).get("dkappa")
        if v is None or not v > 0:
            reasons.append(f"{src}: validation single-colour sub-question dkappa {v} not > 0")
    return reasons


# ================================================================ validation side
def val_c5(c):
    vk, logits, vsha, wsha = p3.load_val_side(c, ("C5",), c.arm_dir)
    return vk, p3.combo_logprobs(("C5",), logits), logits, vsha, wsha


def val_arm(c, x, vk):
    """SW and symmetry-0 validation per-game log-probabilities (0 on unparsed rows, as p03's check)."""
    ddir = os.path.join(c.arm_dir[x], "dumps")
    diff = p3.key_array_diff(os.path.join(ddir, "val_keys.npz"), os.path.join(c.p02_dumps, "val_keys.npz"),
                             ("question_id", "slot", "color", "parsed"))
    if diff:
        raise P04Error(f"arm {x}: val_keys.npz differs from prereg02's {diff}")
    d4 = load_dump(os.path.join(ddir, f"val_p03{x}_d4.npz"), vk["n"], p3.N_SYMS)
    sw = p3.load_swap(os.path.join(ddir, f"val_p03{x}_s0_swap.npz"), vk["n"])
    parsed = vk["parsed"]
    lp_s = log_softmax(np.where(parsed[:, None], sw["logits"], 0.0).astype(np.float64))
    lp_0 = log_softmax(np.where(parsed[:, None], d4["logits"][:, 0, :], 0.0).astype(np.float64))
    return {"d4": d4, "sw": sw, "SW": lp_s, "0": lp_0}


def val_question_inputs(c, vk, build_id):
    groups = vk["groups"]
    cnn_groups = [[i for i in rows if vk["parsed"][i]] for _, rows in groups]
    S = pool_matrix(cnn_groups, vk["n"])
    n = np.array([len(g) for g in cnn_groups], dtype=np.float64)
    op_o, op_has, op = p3.opening_val_aligned(c, vk, build_id)
    mop = np.asarray(pool_matrix([[i for i in rows if op_has[i]] for _, rows in groups], vk["n"]) @ op_o)
    return S, n, mop, op_o, op_has, op


# ================================================================ check
def sw_structure(c, x, keys, vk, c5_val_nan):
    """PREREG_04 2.2 item 1 for one prereg03 arm -> (problems, record)."""
    probs, rec = [], {}
    ddir = os.path.join(c.arm_dir[x], "dumps")
    for stem, names in (("inner_keys", ("rank", "game_id", "player_id", "color", "label")),
                        ("val_keys", ("question_id", "slot", "color", "parsed"))):
        d = p3.key_array_diff(os.path.join(ddir, f"{stem}.npz"), os.path.join(c.p02_dumps, f"{stem}.npz"), names)
        rec[f"{stem}_equal_to_prereg02"] = not d
        if d:
            probs.append(f"arm {x}: {stem}.npz differs from prereg02's ({d})")
    d4, sw = arm_inner(c, x, keys["n"])
    vd4 = load_dump(os.path.join(ddir, f"val_p03{x}_d4.npz"), vk["n"], p3.N_SYMS)
    vsw = p3.load_swap(os.path.join(ddir, f"val_p03{x}_s0_swap.npz"), vk["n"])
    rec["sha256"] = {f"inner_p03{x}_s0_swap.npz": sw["sha256"], f"val_p03{x}_s0_swap.npz": vsw["sha256"],
                     f"inner_p03{x}_d4.npz": d4["sha256"], f"val_p03{x}_d4.npz": vd4["sha256"]}
    wsha = {d["path"]: d["weights_sha256"] for d in (d4, sw, vd4, vsw)}
    rec["weights_sha256"] = d4["weights_sha256"]
    if len(set(wsha.values())) != 1:
        probs.append(f"arm {x}: the dumps name different weights {sorted(set(wsha.values()))}")
    if not c.smoke:
        for side, d in (("inner", sw), ("val", vsw)):
            if d["sha256"] != SW_SHA256[x][side]:
                probs.append(f"arm {x}: {side} swapped dump SHA-256 {d['sha256']} != PREREG_04 1.3 "
                             f"{SW_SHA256[x][side]}")
        if d4["weights_sha256"] != SW_WEIGHTS[x]:
            probs.append(f"arm {x}: weights {d4['weights_sha256']} != prereg03 final.pth {SW_WEIGHTS[x]}")
        cr = read_json(os.path.join(P03_OUT, "check_report.json"))
        want = ((cr.get("sources") or {}).get(p3.ARM_SOURCE[x]) or {}).get("inputs_sha256") or {}
        bad = [k for k, v in rec["sha256"].items() if want.get(k) != v]
        rec["equal_to_prereg03_check_record"] = not bad
        if bad:
            probs.append(f"arm {x}: {bad} differ from prereg03's check_report.json record")
    if not np.isfinite(sw["logits"]).all():
        probs.append(f"arm {x}: non-finite inner swapped logits (NaN pattern must equal C5's: none on inner)")
    if not np.isfinite(d4["logits"]).all():
        probs.append(f"arm {x}: non-finite inner d4 logits")
    nan_sw = np.isnan(vsw["logits"]).all(axis=1)
    fin_sw = np.isfinite(vsw["logits"]).all(axis=1)
    nan_d4 = np.isnan(vd4["logits"]).all(axis=(1, 2))
    if not (np.array_equal(nan_sw, c5_val_nan) and fin_sw[~c5_val_nan].all()):
        probs.append(f"arm {x}: validation swapped logits: NaN rows differ from C5's or non-finite parsed rows")
    if not np.array_equal(nan_d4, c5_val_nan):
        probs.append(f"arm {x}: validation d4 NaN rows differ from C5's")
    rec["val_nan_rows"] = int(nan_sw.sum())
    return probs, rec


def accuracy(lp, label):
    return float(np.mean(np.argmax(lp, axis=1) == label)) * 100


def cmd_check(a):
    t0 = time.time()
    c = resolve(a, "check")
    frozen = start(c)
    commit = committed_inputs(c, val=True)
    p02 = p3.p02_inputs(c, val=True)
    inner = inner_setup(c)
    keys = inner["keys"]
    label = np.asarray(keys["label"])
    vk = load_val_keys(c.p02_dumps)
    c5_val_nan = p3.p02_val_nan(c)
    stops = []
    guard = val_o4_order_guard(c)
    if guard:
        stops.append(guard)
    meta = {}
    lp, sha, probs = inner_sources(c, keys, ("O4", "T4", "SWA", "SWB", "A0", "B0"), meta)
    sources = {"C5": {"status": "usable", "text": SOURCE_TEXT["C5"],
                      "inner_accuracy_percent": accuracy(lp["C5"], label),
                      "weights_sha256": meta["C5_weights_sha256"],
                      "note": "committed prereg02 source (R20); usable by registration"},
               "OP": {"status": "usable", "text": SOURCE_TEXT["OP"], "note": "committed prereg02 opening term"}}
    # ---- 1 structure: O4
    o4 = {"status": "usable", "text": SOURCE_TEXT["O4"], "problems": list(probs), "file": meta.get("O4_file")}
    if c.design:
        o4["pinned_numbers"] = "not checked: DESIGN INPUT preview (no p04_opp.py output yet)"
        o4["dupcheck"] = "not checked: DESIGN INPUT preview"
        o4["selftest"] = "not checked: DESIGN INPUT preview"
    else:
        p, prec = o4_pinned(c, dict(meta["O4_raw"], keys=keys) if "O4_raw" in meta else None)
        o4["problems"] += p
        o4["pinned_numbers"] = prec
        p, drec = dupcheck_status(c)
        o4["problems"] += p
        o4["dupcheck"] = drec
        if c.smoke:
            o4["selftest"] = "not checked: --smoke (synthetic world)"
        else:
            p, strec = selftest_status(c)
            o4["problems"] += p
            o4["selftest"] = strec
    if "O4" in lp:
        o4["inner_accuracy_percent_recorded"] = {"O4": accuracy(lp["O4"], label), "T4": accuracy(lp["T4"], label)}
    if o4["problems"]:
        o4["status"] = "stop"
        stops.extend(o4["problems"])
    sources["O4"] = o4
    # ---- 1 structure: SWA, SWB
    val_arms = {}
    for x in ARMS:
        s = f"SW{x.upper()}"
        p, rec = sw_structure(c, x, keys, vk, c5_val_nan)
        rec.update(text=SOURCE_TEXT[s], problems=p, status="stop" if p else "usable")
        ref = f"{x.upper()}0"
        rec["inner_accuracy_percent_recorded"] = {s: accuracy(lp[s], label), ref: accuracy(lp[ref], label)}
        sources[s] = rec
        stops.extend(p)
        if not p:
            val_arms[x] = val_arm(c, x, vk)
    # ---- 2 colour semantics
    colour = {}
    for src, ref in PAIRS:
        if sources[src]["status"] != "usable":
            colour[src] = {"reference": ref, "status": "not computed (structure stop)"}
            continue
        if src == "O4":
            rec = side_kappa(lp["T4"], lp["O4"], inner)
            rec["val_sub_questions"] = "deferred to apply (needs the validation O4, PREREG_04 2.2 item 2 / 2.6)"
        else:
            x = ARM_OF[src]
            rec = side_kappa(lp[ref], lp[src], inner, vk, val_arms[x]["0"], val_arms[x]["SW"])
            if not c.smoke:
                cr = read_json(os.path.join(P03_OUT, "check_report.json"))
                ref03 = (((cr.get("sources") or {}).get(p3.ARM_SOURCE[x]) or {}).get("colour") or {}) \
                    .get("kappa_single_colour") or {}
                rec["prereg03_record_abs_diff"] = {
                    "inner_dkappa": abs(rec["inner"]["dkappa"] - ((ref03.get("inner") or {}).get("dkappa") or np.nan)),
                    "val_dkappa": abs(rec["val_sub_questions"]["dkappa"]
                                      - ((ref03.get("val_sub_questions") or {}).get("dkappa") or np.nan))}
        reasons = colour_usable(src, rec)
        rec.update(reference=ref, reasons_not_usable=reasons,
                   rule="usable when the single-colour inner dkappa CI lower bound > 0 and (SWA, SWB) the validation "
                        "sub-question dkappa > 0; O4's validation dkappa is required at apply")
        colour[src] = rec
        if reasons:
            sources[src]["status"] = "dropped"
            sources[src]["dropped_because"] = reasons
        sources[src]["colour"] = rec
    usable = [s for s in ("C5", "OP", "O4", "SWA", "SWB") if sources[s]["status"] == "usable"]
    combos = [{"id": cid, "sources": list(srcs), "status": "ok" if set(srcs) <= set(usable) else "dropped",
               "reason": (None if set(srcs) <= set(usable)
                          else f"source(s) {sorted(set(srcs) - set(usable))} not usable")}
              for cid, srcs in COMBOS]
    ms = inner["ms"]
    rep = {"format": CHECK_FORMAT, "what": "PREREG_04 2.2: can O4, SWA, SWB be used (before any fitting)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c),
           "status": "stop" if stops else "ok", "stops": stops, "usable_sources": usable, "sources": sources,
           "combinations": combos, "colour_pairs": [list(p) for p in PAIRS],
           "single_colour_questions": {"questions_sha256": ms["sha256"], "counts": ms["counts"],
                                       "definition": ms["definition"]},
           "val_sub_questions": {"sub_questions": len(p3.val_colour_subquestions(vk)),
                                 "val_questions": len(vk["groups"])},
           "inputs_sha256": dict(sha, **{"val_keys.npz": vk["sha256"]}), "inner_inputs_sha256": inner["inputs"],
           "p02_inputs": p02, "inputs_commit_check": commit, "tool": tool_state(), "frozen_tools": frozen,
           "environment": environment()}
    os.makedirs(c.out_dir, exist_ok=True)
    out = os.path.join(c.out_dir, "check_report.json")
    write_json(out, rep)
    for src, ref in PAIRS:
        r = colour.get(src) or {}
        extra = ""
        if "inner" in r:
            v = r["val_sub_questions"]
            extra = (f": dkappa({ref} - {src}) single-colour inner {r['inner']['dkappa']:+.4f} CI "
                     f"[{r['inner']['ci95'][0]:+.4f}, {r['inner']['ci95'][1]:+.4f}], val "
                     f"{v['dkappa'] if isinstance(v, dict) else v}")
        print(f"{src}: {sources[src]['status']}{extra}")
    print(f"check: usable {usable}; status {rep['status']}{' (preview)' if c.preview else ''}"
          f"{' (DESIGN INPUT)' if c.design else ''} ({time.time() - t0:.0f}s) -> {out}")
    for s in stops:
        print("STOP: " + s, file=sys.stderr)
    return 1 if stops else 0


# ================================================================ selfcheck
def r20_references(c):
    tpath = os.path.join(c.p02_local, "tune_report.json")
    fpath = os.path.join(c.p02_local, "frozen_config.json")
    tr = read_json(tpath) if os.path.exists(tpath) else None
    fc = read_json(fpath) if os.path.exists(fpath) else None
    r20 = next((x for x in (tr or {}).get("combinations", []) if x.get("id") == "R20"), None)
    return tr, fc, r20, tpath, fpath


def q00_p02(inner, smoke, lp5):
    """Q00 = R20 through prereg02's own fit (p03 run_oof / refit_all)."""
    p3.add_combo(inner, "Q00", lp5)
    m, sc, folds, P = p3.run_oof(inner, "Q00", True, smoke, B)
    return m, sc, folds, P


def cmd_selfcheck(a):
    t0 = time.time()
    c = resolve(a, "selfcheck")
    frozen = start(c)
    order = None
    if not c.smoke and c.in_repo:
        order = p3.require_tracked_clean([os.path.join(c.out_dir, "check_report.json")],
                                         "check_report.json (run and commit 'check' before 'selfcheck')")
        cr = read_json(os.path.join(c.out_dir, "check_report.json"))
        if cr.get("format") != CHECK_FORMAT or cr.get("status") != "ok" or cr.get("preview") or \
                cr.get("design_input") or cr.get("smoke"):
            raise P04Error("check_report.json is not a passed real check (status ok, no preview, no design input)")
    p02 = p3.p02_inputs(c, val=True)
    inner = inner_setup(c)
    keys = inner["keys"]
    Xf, Xe, Sf, Se = inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"]
    lp, sha, probs = inner_sources(c, keys, ("O4", "SWA", "SWB"))
    checks, failures = {}, []

    def need(name, ok, detail, required=True):
        checks[name] = {"ok": ok, "detail": detail, "required": required}
        if required and ok is not True:
            failures.append(name)

    lp5 = lp["C5"]
    c5 = p3.load_c5_inner(c, keys["n"])
    need("per_game_logprobs_equal_p02_source_logprobs_C5",
         bool(np.array_equal(lp5, rf.source_logprobs("C5", {p3.C5_DUMP: c5["logits"]}))),
         "combo_logprobs(('C5',)) vs p02_rank_fuse.source_logprobs('C5'), element by element")
    # ---- Q00 = R20 (prereg02's path)
    m, sc, folds, P00 = q00_p02(inner, c.smoke, lp5)
    m.update(inner["boot"].summary(sc))
    theta, refit = p3.refit_all(inner, "Q00", True, c.smoke, B)
    par = params_dict(theta, True, B)
    tr, fc, r20, tpath, fpath = r20_references(c)
    if r20 is not None and r20.get("status") == "ok":
        d_oof = p3.compare_oof(m, r20["oof"])
        d_fold = p3.fold_diffs(folds, r20["folds"])
        need("Q00_inner_oof_equals_R20", not d_oof and not d_fold,
             {"Q00_rank_balanced": m["rank_balanced"], "R20_rank_balanced": r20["oof"]["rank_balanced"],
              "abs_diff": abs(m["rank_balanced"] - r20["oof"]["rank_balanced"]),
              "differences": [list(map(str, d)) for d in (d_oof + d_fold)][:20],
              "compared": "every entry of the R20 OOF record and every fold's params, loss and counts; floats <= "
                          "1e-12, the rest exactly"})
        if not c.smoke:
            need("registered_R20_value", r20["oof"]["rank_balanced"] == R20_INNER_OOF and
                 abs(m["rank_balanced"] - R20_INNER_OOF) <= SELFCHECK_TOL,
                 {"tune_report": r20["oof"]["rank_balanced"], "Q00": m["rank_balanced"], "registered": R20_INNER_OOF})
    else:
        need("Q00_inner_oof_equals_R20", None, "tune_report.json with an R20 entry not available", required=not c.smoke)
    fc_r20 = fc is not None and (fc.get("combo") or {}).get("id") == "R20"
    if fc_r20:
        pd = p3.params_diff(par, fc["params"])
        need("Q00_refit_params_equal_frozen", pd <= SELFCHECK_TOL,
             {"max_abs_diff": pd, "tolerance": SELFCHECK_TOL, "refit_loss": refit.get("loss"),
              "frozen_refit_loss": ((tr or {}).get("refit") or {}).get("loss")})
    else:
        need("Q00_refit_params_equal_frozen", None, "frozen_config.json of R20 not available", required=not c.smoke)
    vpath = os.path.join(c.p02_local, "rank_val_probs.csv")
    pb = os.path.join(c.p02_local, "pred_B.csv")
    if fc_r20 and os.path.exists(vpath):
        vk, _lp5v, vlog, vsha, _w = val_c5(c)
        op_o, op_has, _op = p3.opening_val_aligned(c, vk, fc.get("opening_build_id"))
        P, n, ok = p3.val_probs(("C5",), theta, True, B, vk, vlog, op_o, op_has)
        ref = decide.read_probs(vpath)
        qids = [q for q, _ in vk["groups"]]
        row_bad, max_diff, b_diff = [], 0.0, []
        if [r["question_id"] for r in ref] != qids:
            row_bad.append("question order / set")
        else:
            for k, (r, (q, rows)) in enumerate(zip(ref, vk["groups"])):
                if r["n_games"] != len(rows) or r["n_parsed"] != int(n[k]) or (r["probs"] is None) != (not ok[k]):
                    row_bad.append(q)
                    continue
                if r["probs"] is None:
                    continue
                max_diff = max(max_diff, float(np.max(np.abs(P[k] - np.asarray(r["probs"])))))
                if decide.rule_b([float(x) for x in P[k]]) != decide.rule_b(r["probs"]):
                    b_diff.append(q)
        pb_diff = None
        if os.path.exists(pb):
            with open(pb, newline="", encoding="utf-8") as f:
                saved = {r["question_id"]: r["pred_rank"] for r in csv.DictReader(f)}
            pb_diff = [q for k, q in enumerate(qids)
                       if ok[k] and saved.get(q) != decide.rule_b([float(x) for x in P[k]])]
        need("Q00_val_probs_equal_rank_val_probs", not row_bad and max_diff <= SELFCHECK_TOL and not b_diff
             and not pb_diff,
             {"questions": len(qids), "max_abs_diff": max_diff, "tolerance": SELFCHECK_TOL,
              "row_mismatches": row_bad[:10], "rule_b_differences": b_diff[:10],
              "pred_B_differences": pb_diff if pb_diff is None else pb_diff[:10],
              "inputs_sha256": dict(vsha, **{"rank_val_probs.csv": sha256_file(vpath),
                                            "pred_B.csv": sha256_file(pb) if os.path.exists(pb) else None})})
    else:
        need("Q00_val_probs_equal_rank_val_probs", None, "R20's rank_val_probs.csv / frozen_config.json not available",
             required=not c.smoke)
    # ---- seam: the generalised fit with E = [OP] against prereg02's (same decisions; refit recorded)
    mf, me = pooled(Sf, lp5, "log"), pooled(Se, lp5, "log")
    Pg, gfolds = gen_oof_record(Xf, Xe, mf, me, [Xf.mop], [Xe.mop])
    mg, _ = metrics(rule_b_np(Pg), Xe.y, Xe.n, Xe.player)
    fold_d = [p3.params_diff(as_p02_dict(p02_params(np.asarray(g["theta"]))), f["params"])
              for g, f in zip(gfolds, folds)]
    th_g, rec_g = gen_fit_record(pooled(Sf, lp5, "log"), [Xf.mop], Xf.n, Xf.y)
    seam = {"max_abs_prob_diff_recorded_only": float(np.max(np.abs(Pg - P00))),
            "same_rule_b": bool(np.array_equal(rule_b_np(Pg), rule_b_np(P00))),
            "gen_oof_rank_balanced": mg["rank_balanced"], "p02_oof_rank_balanced": m["rank_balanced"],
            "fold_params_max_abs_diff_recorded_only": max(fold_d),
            "refit_params_max_abs_diff_recorded_only": p3.params_diff(as_p02_dict(p02_params(th_g)), par),
            "refit_loss_diff_recorded_only": abs(rec_g["loss"] - refit["loss"]),
            "required": "the same rule-B decision on every inner eval question and the same OOF rank-balanced mean "
                        "(<= 1e-12): Q00 (prereg02's fit) and Q01 / Q02 (generalised fit) are then compared on equal "
                        "terms",
            "note": "Q00 is taken from prereg02's path (R20 bit for bit); the generalised path's parameters and "
                    "probabilities differ at the 1e-12 level (summation order (m + B beta) + w m_op vs "
                    "(m + w m_op) + B beta), so they are recorded"}
    need("gen_path_Q00_seam", seam["same_rule_b"] and abs(mg["rank_balanced"] - m["rank_balanced"]) <= SELFCHECK_TOL,
         seam)
    # ---- Q01, Q02 (generalised fit) reproduce the design numbers
    design_rec = {}
    if probs:
        need("sources_loaded", False, probs)
    else:
        for cid, srcs in COMBOS[1:]:
            terms = [t for t in TERMS if t in srcs]
            P, gf = gen_oof_record(Xf, Xe, mf, me, question_terms(Xf, Sf, lp, terms),
                                   question_terms(Xe, Se, lp, terms))
            mq, _ = metrics(rule_b_np(P), Xe.y, Xe.n, Xe.player)
            want = DESIGN_OOF[cid]
            design_rec[cid] = {"rank_balanced": mq["rank_balanced"], "design": want,
                               "abs_diff": abs(mq["rank_balanced"] - want),
                               "weights_by_fold": [g["theta"][2:2 + len(terms)] for g in gf], "terms": terms}
            need(f"{cid}_reproduces_design", abs(mq["rank_balanced"] - want) <= SELFCHECK_TOL,
                 design_rec[cid], required=not c.smoke)
    passed = not failures
    rep = {"format": SELFCHECK_FORMAT,
           "what": "PREREG_04 2.3: Q00 reproduces R20 (inner OOF, refit, validation probabilities); Q01 / Q02 "
                   "reproduce the design-phase inner OOF (a check of the port, not a choice)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c),
           "passed": passed, "failures": failures, "checks": checks,
           "Q00": {"oof": m, "folds": folds, "refit": refit, "params": par},
           "design_reproduction": design_rec, "order_check": order, "p02_inputs": p02,
           "inputs_sha256": sha, "inner_inputs_sha256": inner["inputs"],
           "reference_files_sha256": {k: (sha256_file(p) if os.path.exists(p) else None) for k, p in (
               ("tune_report.json", tpath), ("frozen_config.json", fpath), ("rank_val_probs.csv", vpath),
               ("pred_B.csv", pb))},
           "tool": tool_state(), "frozen_tools": frozen, "environment": environment()}
    os.makedirs(c.out_dir, exist_ok=True)
    out = os.path.join(c.out_dir, "selfcheck.json")
    write_json(out, rep)
    for k, v in checks.items():
        state = "ok" if v["ok"] else ("n/a" if v["ok"] is None else "FAILED")
        print(f"  {k}: {state if v['required'] or v['ok'] else state + ' (recorded only)'}")
    for cid, r in design_rec.items():
        print(f"  {cid}: inner OOF {r['rank_balanced']!r} (design {r['design']!r}, |diff| {r['abs_diff']:.3g})")
    print(f"selfcheck: Q00 inner OOF {m['rank_balanced']!r} (R20 {R20_INNER_OOF!r}) -> "
          f"{'PASSED' if passed else 'FAILED'}{' (preview)' if c.preview else ''}"
          f"{' (DESIGN INPUT)' if c.design else ''} ({time.time() - t0:.0f}s) -> {out}")
    return 0 if passed else 1


# ================================================================ tune
def select(results):
    ok = [r for r in results if r["status"] == "ok"]
    best = max(r["oof"]["rank_balanced"] for r in ok)
    within = [r["id"] for r in ok if best - r["oof"]["rank_balanced"] <= SELECT_TOL + FLOAT_EPS]
    chosen = next(r for r in ok if r["id"] == within[0])
    return chosen, {"rule": f"highest inner OOF rank-balanced mean (rule B); within {SELECT_TOL} of the best -> the "
                            f"earliest in the list Q00..Q02 (compared with {FLOAT_EPS} float slack)",
                    "best_id": next(r["id"] for r in ok if r["oof"]["rank_balanced"] == best),
                    "best_rank_balanced": best, "selected": chosen["id"],
                    "selected_rank_balanced": chosen["oof"]["rank_balanced"], "within_tolerance": within}


def gate_record(sel_id, rb_sel, rb_q00, sc_sel, sc_q00, boot, mono_diff):
    """G1-G4 of PREREG_04 2.5 for the selected combination vs Q00 (Q00 itself -> not submitted)."""
    if sel_id == "Q00":
        return {"selected": "Q00", "G1": None, "G2": None, "G3": None, "G4": None, "passed": False,
                "reason": "Q00 (R20) selected: not submitted, R20 unchanged, the scoring slot is kept"}
    delta = rb_sel - rb_q00
    d = boot.stats(sc_sel) - boot.stats(sc_q00)
    lo, hi = ci95(d)
    g3 = float(np.mean(np.asarray(sc_sel) - np.asarray(sc_q00)))
    g1 = {"threshold": GATE, "delta": delta, "passed": bool(delta >= GATE - FLOAT_EPS)}
    g2 = {"ci95": [lo, hi], "se": float(np.std(d, ddof=1)), "passed": bool(lo > 0),
          "bootstrap": f"players within each level, {N_BOOT} draws, default_rng({BOOT_SEED}), paired"}
    g3r = {"question_mean_difference": g3, "passed": bool(g3 >= -FLOAT_EPS),
           "note": "mean of the per-question score differences on the inner eval rounds, >= 0 with 1e-12 slack"}
    bw = {col: (mono_diff["by_colour"].get(col) or {}).get("rank_balanced") for col in p3.MONO_COLOURS}
    g4 = {"rank_balanced_difference": mono_diff["rank_balanced"], "ci95": mono_diff["ci95"],
          "by_colour_point_estimates": bw,
          "passed": bool(mono_diff["ci95"][0] > 0 and all(v is not None and v > 0 for v in bw.values())),
          "questions": mono_diff["n_questions"],
          "rule": "single-colour inner pseudo-questions: paired PlayerBootstrap 95 % CI lower bound > 0 and the black "
                  "and white point estimates > 0"}
    passed = g1["passed"] and g2["passed"] and g3r["passed"] and g4["passed"]
    return {"selected": sel_id, "G1": g1, "G2": g2, "G3": g3r, "G4": g4, "passed": bool(passed),
            "reason": "all four gates passed" if passed else "a gate failed: not submitted, R20 unchanged"}


def require_order(c, names):
    paths = [os.path.join(c.out_dir, n) for n in names]
    missing = [n for n, p in zip(names, paths) if not os.path.exists(p)]
    if missing:
        raise P04Error(f"run the earlier steps first: {missing} missing in {c.out_dir}")
    if c.smoke or c.preview:
        return None
    return p3.require_tracked_clean(paths, f"{names} (commit each step before the next)")


def short_mono(r):
    return {"rank_balanced": r["rank_balanced"], "question_mean": r["question_mean"], "se": r.get("se"),
            "n_questions": r["n_questions"], "per_level": r["per_level"],
            "by_colour": {col: {k: v[k] for k in ("rank_balanced", "question_mean", "n_questions")}
                          for col, v in r["by_colour"].items()},
            "by_k": {k: {kk: v[kk] for kk in ("rank_balanced", "question_mean", "n_questions")}
                     for k, v in r["by_k"].items()}}


def adoption(delta, se):
    if not se or se <= 0:
        return None
    return {f: 0.5 * (1 + math.erf((f * delta / se - 1.96) / math.sqrt(2))) for f in (1.0, 0.75, 0.5)}


def cmd_tune(a):
    t0 = time.time()
    c = resolve(a, "tune")
    frozen_path = os.path.join(c.out_dir, "frozen_config.json")
    if not c.smoke and c.in_repo:
        rc, _ = p3.git(["ls-files", "--error-unmatch", "--", rel(frozen_path)])
        if rc == 0:
            raise P04Error("frozen_config.json is already tracked by git (frozen); tune will not overwrite it")
    frozen = start(c)
    tools = tool_state()
    if not c.smoke and not c.preview:
        require_tools_committed(tools)
        guard = val_o4_order_guard(c)
        if guard:
            raise P04Error(guard)
    order = require_order(c, ["check_report.json", "selfcheck.json"])
    cr = read_json(os.path.join(c.out_dir, "check_report.json"))
    scr = read_json(os.path.join(c.out_dir, "selfcheck.json"))
    if cr.get("format") != CHECK_FORMAT or cr.get("status") != "ok":
        raise P04Error("check_report.json did not pass (status must be ok; a stop needs a fix and a revision)")
    if scr.get("format") != SELFCHECK_FORMAT or scr.get("passed") is not True:
        raise P04Error("selfcheck.json did not pass; tune does not run")
    for rep, what in ((cr, "check_report.json"), (scr, "selfcheck.json")):
        if bool(rep.get("smoke")) != c.smoke or bool(rep.get("preview")) != c.preview or \
                bool(rep.get("design_input")) != c.design:
            raise P04Error(f"{what} is a smoke / preview / design record that does not match this run")
    usable = list(cr["usable_sources"])
    p02 = p3.p02_inputs(c, val=False)   # inner files only
    inner = inner_setup(c)
    if inner["inputs"] != cr["inner_inputs_sha256"] or inner["inputs"] != scr["inner_inputs_sha256"]:
        raise P04Error("inner inputs changed since check / selfcheck")
    keys = inner["keys"]
    need = sorted(({s for _cid, srcs in COMBOS for s in srcs} | {s for r in RECORD_ONLY for s in r[3]}) - {"OP"})
    meta = {}
    lp, sha, probs = inner_sources(c, keys, need, meta)
    if probs:
        raise P04Error(f"inner sources: {probs}")
    for k, v in sha.items():
        if k in cr["inputs_sha256"] and cr["inputs_sha256"][k] != v:
            raise P04Error(f"{k} changed since check")
    commit = committed_inputs(c)
    Xf, Xe, Sf, Se = inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"]
    Xm, Sm, ms, mboot, boot = inner["X_mono"], inner["S_mono"], inner["ms"], inner["mono_boot"], inner["boot"]
    lp5 = lp["C5"]
    mf, me, mm = pooled(Sf, lp5, "log"), pooled(Se, lp5, "log"), pooled(Sm, lp5, "log")
    base = {"C5": (mf, me, mm)}
    if "O4" in lp:
        base["O4"] = (pooled(Sf, lp["O4"], "log"), pooled(Se, lp["O4"], "log"), pooled(Sm, lp["O4"], "log"))
    print(f"inputs ready: fit {len(Xf)}, eval {len(Xe)}, single-colour {len(Xm)} questions; usable {usable} "
          f"({time.time() - t0:.0f}s)", flush=True)

    def run(name, base_src, terms, path):
        """-> (eval OOF P, single-colour P, fold records)."""
        if path == "p02":
            opening = "OP" in terms
            key = "Q00" if opening else "Q00noopen"
            p3.add_combo(inner, key, lp5)
            Xm.m[(key, p3.POOL)] = mm
            _m, _sc, folds, P = p3.run_oof(inner, key, opening, c.smoke, B)
            if not np.array_equal(p3.fold_predict(Xe, key, folds, opening, B), P, equal_nan=True):
                raise P04Error(f"{name}: fold records do not reproduce the OOF probabilities")
            Pm = p3.fold_predict(Xm, key, folds, opening, B)
            for fr in folds:
                th = p3.theta_of(fr["params"], opening)
                fr["theta"] = [float(x) for x in th]
            return P, Pm, folds
        bf, be, bm = base[base_src]
        Ef, Ee, Em = (question_terms(Xf, Sf, lp, terms), question_terms(Xe, Se, lp, terms),
                      question_terms(Xm, Sm, lp, terms))
        P, folds = gen_oof_record(Xf, Xe, bf, be, Ef, Ee)
        if not np.array_equal(gen_fold_predict(folds, be, Ee, Xe.n, Xe.fold), P):
            raise P04Error(f"{name}: fold records do not reproduce the OOF probabilities")
        Pm = gen_fold_predict(folds, bm, Em, Xm.n, Xm.fold)
        for fr in folds:
            fr["params"] = gen_params(fr["theta"], terms)
        return P, Pm, folds

    results, scores, mono_sc, mono_rec, oofP = [], {}, {}, {}, {}
    for cid, srcs in COMBOS:
        rec = {"id": cid, "sources": list(srcs), "terms": [t for t in TERMS if t in srcs],
               "path": "p02" if cid == "Q00" else "gen", "pool": "log", "calibrated": True, "opening": True}
        if not set(srcs) <= set(usable):
            rec.update(status="dropped", reason=f"source(s) {sorted(set(srcs) - set(usable))} not usable (check)")
            results.append(rec)
            continue
        P, Pm, folds = run(cid, "C5", rec["terms"], rec["path"])
        if np.isnan(Pm).any():
            raise P04Error(f"{cid}: a single-colour question without an OOF fold")
        m, sc = metrics(rule_b_np(P), Xe.y, Xe.n, Xe.player)
        m.update(boot.summary(sc, None if cid == "Q00" else scores["Q00"]))
        for k in ("diff_vs_R00_se", "diff_vs_R00_ci95"):
            if k in m:
                m[k.replace("R00", "Q00")] = m.pop(k)
        mono_rec[cid], mono_sc[cid] = p3.mono_score_record(rule_b_np(Pm), Xm, ms, mboot)
        scores[cid], oofP[cid] = sc, P
        if cid != "Q00":
            m["question_mean_diff_vs_Q00"] = float(np.mean(sc - scores["Q00"]))
            m["per_level_diff_vs_Q00"] = {lv: m["per_level"][lv] - results[0]["oof"]["per_level"][lv]
                                          for lv in m["per_level"]}
            se = p3.se_val_record(m["rank_balanced"] - results[0]["oof"]["rank_balanced"], sc - scores["Q00"], Xe.y)
            se["adoption_probability_at_fraction_of_delta"] = adoption(se["delta"], se["se_val"])
            m["se_val_record"] = se
        rec.update(status="ok", oof=m, folds=folds, single_colour=short_mono(mono_rec[cid]),
                   weights_by_fold=[fr["params"]["w"] if isinstance(fr["params"]["w"], dict) else
                                    {"OP": fr["params"]["w"]} for fr in folds])
        results.append(rec)
        print(f"{cid} {'+'.join(srcs):22s}: OOF rank-balanced {m['rank_balanced']!r}, question mean "
              f"{m['question_mean']:.4f}; single-colour {mono_rec[cid]['rank_balanced']:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
    if results[0]["status"] != "ok":
        raise P04Error("Q00 dropped: C5 and OP are usable by registration")
    q00 = results[0]
    if abs(q00["oof"]["rank_balanced"] - scr["Q00"]["oof"]["rank_balanced"]) > SELFCHECK_TOL:
        raise P04Error("Q00 differs from the selfcheck's Q00")
    for cid in DESIGN_OOF:
        r = next(x for x in results if x["id"] == cid)
        d = scr.get("design_reproduction", {}).get(cid)
        if r["status"] == "ok" and d is not None and \
                abs(r["oof"]["rank_balanced"] - d["rank_balanced"]) > SELFCHECK_TOL:
            raise P04Error(f"{cid} differs from the selfcheck's {cid}")
    # ---- record only
    record_only = []
    for name, bsrc, terms, needs, path in RECORD_ONLY:
        missing = [s for s in needs if s not in ("OP", "C6A") and s not in usable]
        if missing or any(s not in lp for s in needs if s not in ("OP",)):
            record_only.append({"name": name, "status": "skipped", "reason": f"source(s) {missing} not usable"})
            continue
        P, Pm, folds = run(name, bsrc, list(terms), path)
        mr, scr_ = metrics(rule_b_np(P), Xe.y, Xe.n, Xe.player)
        d = boot.stats(scr_) - boot.stats(scores["Q00"])
        monor, msc = p3.mono_score_record(rule_b_np(Pm), Xm, ms, mboot)
        record_only.append({"name": name, "status": "ok", "base": bsrc, "terms": list(terms), "path": path,
                            "rank_balanced": mr["rank_balanced"], "question_mean": mr["question_mean"],
                            "per_level": mr["per_level"], "per_n_bin": mr["per_n_bin"],
                            "diff_vs_Q00": mr["rank_balanced"] - q00["oof"]["rank_balanced"],
                            "diff_vs_Q00_ci95": ci95(d),
                            "question_mean_diff_vs_Q00": float(np.mean(scr_ - scores["Q00"])),
                            "single_colour": short_mono(monor),
                            "single_colour_diff_vs_Q00": p3.mono_diff_record(ms, mboot, msc, mono_sc["Q00"]),
                            "weights_by_fold": [fr["params"]["w"] for fr in folds]})
        dd = mr['rank_balanced'] - q00['oof']['rank_balanced']
        print(f"record only {name:22s}: OOF {mr['rank_balanced']:.4f} ({dd:+.4f} vs Q00); single-colour "
              f"{monor['rank_balanced']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    # ---- selection, gates, refit
    chosen, selection = select(results)
    cid = chosen["id"]
    mono_diffs = {r["id"]: p3.mono_diff_record(ms, mboot, mono_sc[r["id"]], mono_sc["Q00"])
                  for r in results if r["status"] == "ok" and r["id"] != "Q00"}
    for r in results:
        if r["id"] in mono_diffs:
            r["single_colour_diff_vs_Q00"] = mono_diffs[r["id"]]
    gates = gate_record(cid, chosen["oof"]["rank_balanced"], q00["oof"]["rank_balanced"], scores[cid], scores["Q00"],
                        boot, mono_diffs.get(cid))
    if chosen["path"] == "p02":
        theta, refit = p3.refit_all(inner, "Q00", True, c.smoke, B)
        params = p02_params(theta)
    else:
        bf = question_terms(Xf, Sf, lp, chosen["terms"])
        theta, refit = gen_fit_record(mf, bf, Xf.n, Xf.y)
        refit["fit_questions"] = len(Xf)
        params = gen_params(theta, chosen["terms"])
    blocked = []
    if cid == "Q00":
        blocked.append("Q00 (R20) selected: nothing new to score, R20 unchanged, the scoring slot is kept")
    elif not gates["passed"]:
        blocked.append("gates not passed (PREREG_04 2.5): not submitted, R20 unchanged, the scoring slot is kept")
    pending = ["O4: validation single-colour sub-question dkappa (T4 - O4) > 0, computed by apply"] \
        if "O4" in chosen["sources"] else []
    score_allowed = not blocked
    kref = {s: (cr["sources"].get(s) or {}).get("colour") for s in chosen["sources"] if s in ("O4", "SWA", "SWB")}
    src_sha = {"C5": {"inner_p02e9_d4.npz": sha.get(f"inner_{p3.C5_DUMP}.npz"),
                      "weights_sha256": meta["C5_weights_sha256"]},
               "OP": {"opening_inner.npz": inner["inputs"]["opening_inner.npz"],
                      "opening_build_id": inner["op"]["meta"]["build_id"]}}
    if "O4" in chosen["sources"]:
        src_sha["O4"] = {"inner": sha.get("O4_inner"), "design_input": c.design,
                         "count_report": (cr["sources"]["O4"].get("pinned_numbers") or {}).get("count_report")
                         if isinstance(cr["sources"]["O4"].get("pinned_numbers"), dict) else None,
                         "counts_npz": (cr["sources"]["O4"].get("pinned_numbers") or {}).get("counts_npz")
                         if isinstance(cr["sources"]["O4"].get("pinned_numbers"), dict) else None,
                         "dupcheck": cr["sources"]["O4"].get("dupcheck"),
                         "selftest": cr["sources"]["O4"].get("selftest"),
                         "tables_sha256": ((meta.get("O4_file") or {}).get("report") or {}).get("tables_sha256"),
                         "inner_report_p04_opp_sha256": ((meta.get("O4_file") or {}).get("report") or {})
                         .get("p04_opp_sha256"),
                         "p04_opp_tool_sha256": tools["files_sha256"].get(OPP_TOOL)}
    for s in ("SWA", "SWB"):
        if s in chosen["sources"]:
            x = ARM_OF[s]
            src_sha[s] = {"inner": sha.get(f"inner_p03{x}_s0_swap.npz"),
                          "val_pinned": SW_SHA256[x]["val"], "weights_sha256": cr["sources"][s].get("weights_sha256")}
    inputs = dict(inner["inputs"], **sha)
    inputs["check_report.json"] = p3.sha256_file(os.path.join(c.out_dir, "check_report.json"))
    inputs["selfcheck.json"] = p3.sha256_file(os.path.join(c.out_dir, "selfcheck.json"))
    se_sel = chosen["oof"].get("se_val_record")
    report = {
        "format": TUNE_FORMAT,
        "what": "PREREG_04 competition 2 tune: Q00..Q02 on the inner pseudo-questions, 5-fold OOF, gates G1-G4",
        "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c),
        "definitions": {"sources": SOURCE_TEXT, "terms_order": list(TERMS),
                        "formula": "p = softmax(e^a (n/12)^gamma (m_C5 + sum_j w_j E_j + b)), E_j = mean over the "
                                   "question's parsed games of log_softmax(per-game 10-vector), b = B beta",
                        "objective": f"level-weighted NLL + {rf.PENALTY} ||beta||^2, L-BFGS-B bounds {rf.BOUNDS} "
                                     f"(every w), starts a in {rf.STARTS}, maxiter {rf.MAXITER}",
                        "paths": "Q00 through prereg02's fit (R20 bit for bit); the others through gen_fit",
                        "oof": "prereg02: fold f fits on the fit rounds of the other folds' players, predicts fold f's "
                               "eval rounds (and fold f's single-colour pseudo-questions)",
                        "bootstrap": f"players within each level, {N_BOOT} draws, default_rng({BOOT_SEED}), paired "
                                     f"with Q00"},
        "usable_sources": usable, "combinations": results, "record_only": record_only, "selection": selection,
        "gates": gates, "score_allowed": score_allowed, "score_blocked_reasons": blocked,
        "score_pending_at_apply": pending, "se_val": se_sel,
        "refit": dict(refit, combo=cid, params=params), "fail_opening": FAIL_OPENING,
        "questions": {"fit": len(Xf), "eval": len(Xe), "single_colour": len(Xm)},
        "inputs_sha256": inputs, "p02_inputs": p02, "order_check": order, "inputs_commit_check": commit,
        "tool": tools, "frozen_tools": frozen,
        "environment": environment()}
    frozen_obj = {
        "format": FROZEN_FORMAT,
        "what": "PREREG_04 competition 2 frozen configuration (selected on inner; commit before p04_opp.py val and "
                "apply)",
        "created": report["created"], **mode_flags(c),
        "combo": {"id": cid, "sources": list(chosen["sources"]), "terms": chosen["terms"], "path": chosen["path"],
                  "pool": "log", "calibrated": True, "opening": True},
        "source_definitions": {s: SOURCE_TEXT[s] for s in chosen["sources"]}, "source_sha256": src_sha,
        "params": params, "folds": [{"fold": fr["fold"], "loss": fr.get("loss"),
                                     "params": (fr["params"] if chosen["path"] == "gen"
                                                else p02_params(np.asarray(fr["theta"], dtype=np.float64)))}
                                    for fr in chosen["folds"]],
        "basis": [[float(x) for x in row] for row in B], "formula": report["definitions"]["formula"],
        "decision_rule": "B, tools/decide.py (argmax p_k + e^-1 (p_{k-1} + p_{k+1}))",
        "opening_build_id": inner["op"]["meta"]["build_id"],
        "selection": selection, "gates": gates, "score_allowed": score_allowed, "score_blocked_reasons": blocked,
        "score_pending_at_apply": pending, "se_val": se_sel,
        "oof": {k: chosen["oof"][k] for k in ("rank_balanced", "question_mean", "per_level", "per_n_bin",
                                             "pred_distribution")},
        "Q00_oof_rank_balanced": q00["oof"]["rank_balanced"],
        "single_colour": {"questions_sha256": ms["sha256"], "selected": short_mono(mono_rec[cid]),
                          "Q00": short_mono(mono_rec["Q00"]), "selected_minus_Q00": mono_diffs.get(cid),
                          "kappa_reference": kref,
                          "note": "kappa references for the test-day colour check (PREREG_04 5.2): dkappa = kappa("
                                  "target-side reference) - kappa(opponent-side source)"},
        "fail_opening": FAIL_OPENING,
        "check": {"usable_sources": usable, "check_report_sha256": inputs["check_report.json"],
                  "selfcheck_sha256": inputs["selfcheck.json"]},
        "inputs_sha256": inputs, "tool": tools, "environment": environment()}
    os.makedirs(c.out_dir, exist_ok=True)
    write_json(os.path.join(c.out_dir, "tune_report.json"), report)
    write_json(frozen_path, frozen_obj)
    print(f"selected {cid} (OOF {chosen['oof']['rank_balanced']:.4f}; best {selection['best_id']} "
          f"{selection['best_rank_balanced']:.4f}; Q00 {q00['oof']['rank_balanced']:.4f}); gates "
          f"{'PASSED' if gates['passed'] else 'not passed'} ({gates['reason']}); score allowed at tune {score_allowed}"
          f"{' (pending at apply: ' + '; '.join(pending) + ')' if pending and score_allowed else ''}")
    if gates.get("G4"):
        g = gates["G4"]
        print(f"G4 single-colour: {g['rank_balanced_difference']:+.4f} CI [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}], "
              f"B {g['by_colour_point_estimates']['B']:+.4f}, W {g['by_colour_point_estimates']['W']:+.4f}")
    print(f"wrote tune_report.json and frozen_config.json in {c.out_dir}"
          f"{' (PREVIEW; never applied)' if c.preview else '; commit frozen_config.json before apply'}")
    return 0


# ================================================================ apply
def check_report_binding(c, fc, crp):
    """apply's check_report.json: in real mode committed and unchanged; in every mode the file tune recorded
    (frozen_config.json check.check_report_sha256) and a passed check of this mode. -> (report, sha256, commit)."""
    commit = None
    if not c.smoke:
        commit = p3.require_tracked_clean([crp], "check_report.json (the committed check tune used)")
    crsha = sha256_file(crp)
    want = (fc.get("check") or {}).get("check_report_sha256")
    if crsha != want:
        raise P04Error(f"check_report.json SHA-256 {crsha[:12]} differs from the one tune recorded "
                       f"({str(want)[:12]}; frozen_config.json check.check_report_sha256)")
    cr = read_json(crp)
    if cr.get("format") != CHECK_FORMAT or cr.get("status") != "ok" or bool(cr.get("smoke")) != c.smoke or \
            cr.get("preview") or cr.get("design_input"):
        raise P04Error("check_report.json is not a passed check of this mode (status ok, no preview / design input)")
    return cr, crsha, commit


def val_report_binding(c, fc, o4rec, frozen_sha):
    """apply's validation O4: val_report.json names this frozen_config.json (SHA-256), the p04_opp.py tune recorded
    and the rate tables tune froze (tables_sha256). Real mode: all three must be present and equal; smoke: the tables
    when the frozen config has them. -> record."""
    vrep = o4rec.get("report") or {}
    want_tab = ((fc.get("source_sha256") or {}).get("O4") or {}).get("tables_sha256")
    want_tool = (((fc.get("tool") or {}).get("files_sha256")) or {}).get(OPP_TOOL)
    bind = {"frozen_config_sha256": (vrep.get("frozen_config_sha256"), frozen_sha),
            "p04_opp_sha256": (vrep.get("p04_opp_sha256"), want_tool),
            "tables_sha256": (vrep.get("tables_sha256"), want_tab)}
    rec = {k: {"val_report": g, "expected": w, "equal": g == w} for k, (g, w) in bind.items()}
    if not c.smoke:
        bad = sorted(k for k, (g, w) in bind.items() if g is None or w is None or g != w)
        if bad:
            raise P04Error(f"validation O4: val_report.json does not match this frozen config / tool / tables ({bad}): "
                           f"it must come from p04_opp.py val run on the committed frozen_config.json with the tool "
                           f"and rate tables tune recorded")
    elif want_tab is not None and vrep.get("tables_sha256") != want_tab:
        raise P04Error("validation O4: tables_sha256 differs from the frozen one (smoke)")
    return rec


def cmd_apply(a):
    t0 = time.time()
    c = resolve(a, "apply")
    fpath = os.path.join(c.out_dir, "frozen_config.json")
    commit = None
    if not c.smoke:
        commit = p3.require_tracked_clean([fpath], "frozen_config.json (commit it before apply)")
        rc, _ = p3.git(["diff", "--quiet", "HEAD", "--", rel(fpath)])
        if rc != 0:
            raise P04Error("frozen_config.json differs from HEAD; apply only runs on the committed frozen config")
    elif not os.path.exists(fpath):
        raise P04Error("run tune first (frozen_config.json missing)")
    frozen = start(c)
    fc = read_json(fpath)
    if fc.get("format") != FROZEN_FORMAT:
        raise P04Error(f"{fpath}: not a {FROZEN_FORMAT} file")
    if bool(fc.get("smoke")) != c.smoke or fc.get("preview") or fc.get("design_input"):
        raise P04Error("frozen_config.json is a smoke / preview / design-input record that does not match this run")
    Bf = np.asarray(fc["basis"], dtype=np.float64)
    if Bf.shape != (L, L - 1) or not np.array_equal(Bf, B):
        raise P04Error("frozen basis differs from the fixed basis")
    tools = tool_state()
    if not c.smoke:
        changed = [f for f, h in tools["files_sha256"].items() if fc["tool"]["files_sha256"].get(f) != h]
        if changed:
            raise P04Error(f"tool files changed since tune: {changed}; rerun tune (and revise) first")
    combo = fc["combo"]
    srcs = tuple(combo["sources"])
    if (combo["id"], srcs) not in COMBOS or combo.get("terms") != [t for t in TERMS if t in srcs]:
        raise P04Error(f"frozen combination {combo} is not one of Q00..Q02")
    par = fc["params"]
    theta = np.asarray(par["theta"], dtype=np.float64)
    if not np.allclose(B @ np.asarray(par["beta"]), par["b"], atol=1e-12) or \
            list(theta[2 + len(combo["terms"]):]) != par["beta"]:
        raise P04Error("frozen parameters are inconsistent (b != B beta or theta != beta)")
    crp = os.path.join(c.out_dir, "check_report.json")
    if not os.path.exists(crp):
        raise P04Error("check_report.json missing")
    cr, crsha, check_commit = check_report_binding(c, fc, crp)
    inputs_commit = committed_inputs(c, val=True)
    p02 = p3.p02_inputs(c, val=True)
    vk, lp5, vlog, vsha, wsha = val_c5(c)
    if wsha.get(p3.C5_DUMP) != fc["source_sha256"]["C5"]["weights_sha256"]:
        raise P04Error("val_p02e9_d4.npz names other weights than the tuned inner dump")
    lpv = {"C5": lp5}
    inputs = dict(vsha, **{"val_keys.npz": vk["sha256"], "frozen_config.json": sha256_file(fpath)})
    val_colour = {}
    blocked = []
    for s in ("SWA", "SWB"):
        if s in srcs:
            x = ARM_OF[s]
            va = val_arm(c, x, vk)
            if not c.smoke and va["sw"]["sha256"] != SW_SHA256[x]["val"]:
                raise P04Error(f"{s}: val swapped dump SHA-256 {va['sw']['sha256']} != PREREG_04 1.3")
            if va["sw"]["sha256"] != (cr["sources"][s].get("sha256") or {}).get(f"val_p03{x}_s0_swap.npz"):
                raise P04Error(f"{s}: val swapped dump changed since check")
            lpv[s] = va["SW"]
            inputs[f"val_p03{x}_s0_swap.npz"] = va["sw"]["sha256"]
            v = ((cr["sources"][s].get("colour") or {}).get("val_sub_questions") or {}).get("dkappa")
            val_colour[s] = {"val_sub_question_dkappa_from_check": v}
            if v is None or not v > 0:
                blocked.append(f"{s}: validation single-colour sub-question dkappa {v} not > 0 (check)")
    o4rec = None
    if "O4" in srcs:
        probs, o4rec, ov = load_o4_val(c, vk)
        if probs:
            raise P04Error(f"validation O4: {probs} (replay failures must be 0; stop and find out why)")
        o4rec["bound_to_frozen_config"] = val_report_binding(c, fc, o4rec, inputs["frozen_config.json"])
        lpv["O4"] = log_softmax(ov["O4"])
        lpv_t4 = log_softmax(ov["T4"])
        inputs["O4_val"] = o4rec["sha256"]
        kv = val_side_kappa(vk, lpv_t4, lpv["O4"])
        val_colour["O4"] = kv
        if kv.get("dkappa") is None or not kv["dkappa"] > 0:
            blocked.append(f"O4: validation single-colour sub-question dkappa (T4 - O4) {kv.get('dkappa')} not > 0 "
                           f"(PREREG_04 2.2 item 2 / 2.6): not submitted, R20 unchanged, the scoring slot is kept")
    questions = read_question_meta(c.questions)
    by_q = {q: (n, cols) for q, n, cols in questions}
    groups = vk["groups"]
    qids = [q for q, _ in groups]
    if c.smoke:
        if qids != [q for q, _, _ in questions][:len(qids)]:
            raise P04Error("smoke val questions are not the first questions of the question file")
    elif qids != [q for q, _, _ in questions]:
        raise P04Error("val_keys questions differ from the question file (set or order)")
    for q, rows in groups:
        n_, cols = by_q[q]
        if len(rows) != n_ or [vk["color"][i] for i in rows] != cols:
            raise P04Error(f"question {q}: val_keys slots / colours differ from the question file")
    S, n, mop, op_o, op_has, op = val_question_inputs(c, vk, fc["opening_build_id"])
    inputs.update({"opening_val.npz": op["sha256"], "questions": sha256_file(c.questions)})
    ok = n > 0
    P = np.full((len(groups), L), np.nan)
    if combo["path"] == "p02":
        P, n2, ok2 = p3.val_probs(("C5",), theta, True, B, vk, vlog, op_o, op_has)
        if not (np.array_equal(n2, n) and np.array_equal(ok2, ok)):
            raise P04Error("prereg02 path question sizes differ")
    else:
        m = pooled(S, lp5, "log")
        Es = [mop if t == "OP" else pooled(S, lpv[t], "log") for t in combo["terms"]]
        if ok.any():
            z, _ = gen_logits(theta, m[ok], [E[ok] for E in Es], n[ok])
            P[ok] = softmax(z)
    os.makedirs(c.out_dir, exist_ok=True)
    out_csv = os.path.join(c.out_dir, "rank_val_probs.csv")
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(PROB_COLUMNS)
        for k, (q, rows) in enumerate(groups):
            probs_k = [repr(float(x)) for x in P[k]] if ok[k] else [""] * L
            w.writerow([q, len(rows), int(n[k])] + probs_k)
    os.replace(tmp, out_csv)
    decide.read_probs(out_csv)   # must pass decide.py's strict reader
    gates = fc.get("gates")
    if not isinstance(gates, dict) or gates.get("passed") is not True:
        blocked.insert(0, "gates not passed in tune (PREREG_04 2.5): do not score; record 'not submitted'")
    if combo["id"] == "Q00":
        blocked.insert(0, "Q00 (R20) selected: nothing new to score")
    rep = {"format": APPLY_FORMAT,
           "what": "PREREG_04 competition 2 apply: frozen configuration on the validation questions",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c),
           "commit_check": commit, "inputs_commit_check": inputs_commit,
           "check_report": {"sha256": crsha, "equal_to_frozen_config_record": True, "commit_check": check_commit},
           "combo": combo, "params": par,
           "questions": len(groups), "questions_without_parsed_game": int((~ok).sum()),
           "games": vk["n"], "games_parsed": int(vk["parsed"].sum()),
           "val_colour_check": val_colour, "o4_val_file": o4rec,
           "inputs_sha256": inputs, "p02_inputs": p02, "output": os.path.abspath(out_csv),
           "output_sha256": sha256_file(out_csv), "tool": tools, "frozen_tools": frozen, "environment": environment(),
           "gates": gates, "score_allowed": not blocked, "score_blocked_reasons": blocked,
           "next": "PY tools/decide.py results/prereg04/rank/rank_val_probs.csv results/prereg04/rank; PY "
                   "tools/check_submission.py rank results/prereg04/rank/pred_B.csv --questions "
                   "data/local/rank_val_questions.csv; commit; H4R (PREREG_04 3) only when score_allowed is true"}
    write_json(os.path.join(c.out_dir, "apply_report.json"), rep)
    print(f"apply {combo['id']}: {len(groups)} questions ({int((~ok).sum())} without a parsed game) -> {out_csv}")
    if "O4" in val_colour:
        print(f"O4 validation single-colour dkappa (T4 - O4): {val_colour['O4'].get('dkappa')}")
    print("score allowed: %s%s" % (not blocked, (" (" + "; ".join(blocked) + ")") if blocked else ""))
    print(f"next: PY tools/decide.py {out_csv} {c.out_dir}")
    return 0


# ================================================================ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["check", "selfcheck", "tune", "apply"])
    ap.add_argument("--p02-dir", help="prereg02 rank result dir with dumps/ (real: results/prereg02/rank)")
    ap.add_argument("--p02-local-dir", help="pseudo_inner.json, opening_*.npz and R20's tune/frozen/probs files "
                                            "(real: results/prereg02/rank)")
    ap.add_argument("--p03-dir", help="folder holding rank_a/ and rank_b/ (real: results/prereg03)")
    ap.add_argument("--opp-dir", help="p04_opp.py outputs (real: results/prereg04/opp)")
    ap.add_argument("--out-dir", help="real: results/prereg04/rank (or outside the repository for a preview of check / "
                                      "selfcheck / tune); --smoke: required, outside the repository")
    ap.add_argument("--split", help="players_split.csv (real: results/split_v1/players_split.csv)")
    ap.add_argument("--questions", help="apply: question file (real: data/local/rank_val_questions.csv)")
    ap.add_argument("--design-o4", action="store_true",
                    help="preview only: O4 / T4 inner vectors from the design file proxy_inner_all.npz (flagged)")
    ap.add_argument("--smoke", action="store_true", help="synthetic inputs; output outside the repository")
    a = ap.parse_args(argv)
    try:
        return {"check": cmd_check, "selfcheck": cmd_selfcheck, "tune": cmd_tune, "apply": cmd_apply}[a.command](a)
    except ERRORS as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
