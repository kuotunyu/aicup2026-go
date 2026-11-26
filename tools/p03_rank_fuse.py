"""PREREG_03 competition 2: recipe-v2 arms fused with R20 in log space (PREREG_03 2; plan.md 4).

    PY = .venv/Scripts/python.exe
    PY tools/p03_rank_fuse.py unpack --arm a|b ZIP [--dest D --smoke]
    PY tools/p03_rank_fuse.py check      [--out-dir O]                  -> O/check_report.json
    PY tools/p03_rank_fuse.py selfcheck  [--out-dir O]                  -> O/selfcheck.json
    PY tools/p03_rank_fuse.py tune                                      -> tune_report.json, frozen_config.json
    PY tools/p03_rank_fuse.py apply      [--questions Q]                -> rank_val_probs.csv, apply_report.json
    then: PY tools/decide.py results/prereg03/rank/rank_val_probs.csv results/prereg03/rank
    smoke (notebook SMOKE dumps or synthetic data; every path given, output outside the repository):
    PY tools/p03_rank_fuse.py <cmd> --smoke --p02-dir D --p02-local-dir L --p03-dir P --out-dir O --split S
                                     [--fallback-r10 F] [--questions Q] [--min-acc X] [--smoke-record-colour]
    (--min-acc and --smoke-record-colour exist for notebook SMOKE dumps, whose barely trained models sit far
    below the 24.15 % floor and whose colour signal is noise; both are refused without --smoke)

Real mode reads only the repository paths: prereg02's committed R20 inputs in results/prereg02/rank
(dumps/, pseudo_inner.json, opening_inner.npz, opening_val.npz; never recomputed), the unpacked arms
results/prereg03/rank_a/ and rank_b/, results/split_v1/players_split.csv and
results/test02/rank/fallback_R10.json; it writes only into results/prereg03/rank (an --out-dir outside the
repository gives a preview of check / selfcheck, which tune never accepts). Nothing is ever written into
results/prereg01, results/prereg02, results/test02 or the arm folders. No validation label file is read.

Every command first compares tools/p02_rank_fuse.py, p02_rank_pseudo.py, p02_rank_opening.py, p02_moves.py,
c10data.py and decide.py with their HEAD blobs (SHA-256 with CRLF folded to LF) and refuses when one differs;
real mode also requires them equal to the hashes recorded in results/prereg02/rank/frozen_config.json.

unpack    One arm's result zip (prereg03_rank_{a,b}_results.zip; members under rank_{a,b}/) into
          results/prereg03/rank_{a,b}/ plus results/prereg03/unpack_rank_{a,b}.json (zip SHA-256). Refuses
          weights (.pth .pt ...), temporary files, a *failed* zip, a missing required file (ARM_REQUIRED), a
          dump whose completion marker SHA-256 differs, a run whose last run_log status is not 'completed',
          an arm folder that ever recorded a registered stop (any run_log run with status in STOP_STATUSES,
          config.json 'stopped', or STOPPED.json; PREREG_03 1.3 "any stop: that arm is not a result"), config.json
          smoke not false (not true with --smoke) and an existing target. Commit afterwards.
check     Per arm (PREREG_03 2.2, plan.md 4.4), before any combination is fitted:
          1 structure: inner/val key arrays equal to prereg02's array by array; the four dumps' weights
            SHA-256 equal to each other, to their _DONE markers (marker SHA-256 = file, weights final.pth)
            and to config.json final_sha256; config.json training_complete true, recipe.T = T, recipe.ARM =
            the arm; run_log's last run 'completed' and no recorded stop (as unpack); history.csv's largest
            global_step = T (33,768); inner
            logits finite; validation logits: shape, NaN pattern equal to val_p02e9_d4, SHA-256 recorded;
            readonly_after.json's last run says unchanged.
          2 floor: inner per-game accuracy of the original orientation < 24.15 % -> source unusable.
          3 colour wiring: accuracy(colour swapped) - accuracy(correct) > 0.2 points -> stop (bug).
          4 colour semantics: kappa = per question, mean over its games g of
            sum_k softmax(mean_{h != g} l_h)_k * l_g,k (l = original-orientation log_softmax), then the plain
            mean over questions (questions with >= 2 parsed games). dkappa = kappa(correct) - kappa(whole
            question swapped). On the inner eval rounds with a PlayerBootstrap 95 % CI (players resampled
            within each level, N_BOOT draws, default_rng(BOOT_SEED); statistic = the plain question mean, as
            kappa). When dacc (correct - swapped) >= 0.5 points and the inner CI lower bound > 0, the
            validation dkappa must be > 0, else stop. Recorded otherwise; never used for the selection.
          5 recorded: per-symmetry accuracy (and C5's), history.csv rows (learning curve, B's accepted games).
          6 recorded only (PREREG_03 rev 3), the single-colour kappa reference: dkappa (same definition) on the
            single-colour inner pseudo-questions (below) with a PlayerBootstrap 95 % CI, split by colour (each
            split draws its own players) and by k; and on the validation questions' single-colour sub-questions
            (each question's parsed games split by the question file's colour column, val_keys.npz; no label;
            >= 2 parsed games), split by colour, with a question-level bootstrap CI recorded; the test-day
            condition dacc >= 0.5 points and single-colour inner CI lower bound > 0. Never changes a stop or the
            usable sources (tune reads it for score_allowed, PREREG_03 rev 3 11(c)).
          Writes check_report.json (usable sources; status 'stop' and exit 1 when any stop fired).
selfcheck P00 (C5 alone = R20) through this tool's own path must reproduce prereg02: the 5-fold inner OOF
          record of R20 in tune_report.json (rank-balanced 0.5854702521055817, question mean, every level,
          every games-per-question bin, predicted distributions, bootstrap SE, every fold's parameters and
          loss; <= 1e-12, counts equal), the per-game log-probabilities equal to p02's
          source_logprobs('C5') element by element, the refit on all fit questions vs frozen_config.json
          params (<= 1e-12), and the validation probabilities vs rank_val_probs.csv row by row (<= 1e-12, same
          n_games / n_parsed / blanks) with rule B identical (decide.rule_b on both and pred_B.csv).
tune      Needs (real mode) committed tools, committed arm results, committed check_report.json (status ok)
          and committed selfcheck.json (passed, not a preview), and no tracked frozen_config.json. Reads
          inner files only (never a val_* file). Combinations P00..P06 in the fixed order (PREREG_03 2.3);
          those using an unusable source are dropped. 5-fold OOF on the eval rounds; selection = highest OOF
          rank-balanced mean, anything within 0.003 of it -> the earliest. Gates on the selected one only:
          G1 selected - P00 >= 0.010; G2 paired PlayerBootstrap 95 % CI lower bound > 0; G3 mean per-question
          difference >= 0 (all with 1e-12 float slack). Recorded: SE_val = sqrt(sum_l Var_l(d) / N_l) / 10 (d
          the per-question score difference on the inner eval rounds, ddof 1, N_l the validation questions per
          level), adoption probability Phi(delta / SE_val - 1.96), per-level differences, per-bin predicted
          distributions. Refit of the selected on all fit questions; its no-opening version (w = 0, same
          procedure, OOF and refit); FAIL_OPENING choice = the no-opening version when its inner OOF is higher
          than R10's (fallback_R10.json, 0.5687), else R10. Recorded only (PREREG_03 rev 3): every fitted
          combination scored on the single-colour inner pseudo-questions with its 5 fold records' parameters
          (checked first to reproduce the eval-round OOF probabilities exactly), all / black / white / per k, and
          the selected minus P00 with a paired PlayerBootstrap CI; frozen_config.json keeps the selected, P00, the
          difference and the selected sources' single-colour kappa reference from check_report.json.
          score_allowed = gates passed, the selection is not P00, and no C6 source of the selection has the
          single-colour condition with a validation single-colour sub-question dkappa not > 0 (PREREG_03 rev 3
          11(c): such a source cannot be used on test day, so H3R is not run and the scoring chance is kept;
          score_blocked_single_colour lists why). Writes tune_report.json, frozen_config.json.
apply     Needs frozen_config.json committed and identical to HEAD, the arm results committed and unchanged,
          and the tools the ones tune recorded. Reads the validation dumps of the selected sources (parse sets
          must be identical), opening_val.npz and the question file; writes rank_val_probs.csv (prereg01
          column layout, probabilities as repr, blank when n_parsed = 0) and apply_report.json with the gates
          and score_allowed (= gates passed, the selection is not P00 and no single-colour block, recomputed from
          frozen_config.json's single_colour.kappa_reference as tune does).

Sources (plan.md 4.2; dumps [N, 8, 10] fp32 logits): C5 = prereg02 epoch 9 (p02e9_d4, committed),
C6A = arm A final (p03a_d4), C6B = arm B final (p03b_d4). A combination concatenates its sources'
orientation lists and takes log_softmax(mean of the 8k log_softmax arrays): equal weight per source, no new
parameter; P00 is exactly p02's C5. Question model, fit, OOF, bootstrap and rule B are p02_rank_fuse's
(imported, never edited): p = softmax(e^a (n/12)^gamma (m + w m_op + b)), log pooling, opening term.

Single-colour inner pseudo-questions (PREREG_03 rev 3; official competition 2 questions hold 5/10/15/20 games of one
colour, the registered inner pseudo-questions and the local validation mix colours): every bank player of
pseudo_inner.json (its fold) and colour c; pool = all its dumped inner games of colour c ordered by
sha256('c10-prereg03|mono|order|{rank}|{game_id}'); rounds 0..31, k in 5/10/15/20 with k <= pool size, positions
random.Random(int(sha256('c10-prereg03|mono|q|{r}|{player_id}|{c}|{k}')[:16], 16)).sample(range(pool), k). Inner
files only; recorded only.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import zipfile

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import c10data  # noqa: E402
import decide  # noqa: E402
import p02_moves  # noqa: E402,F401  (hash-checked with the other frozen tools)
import p02_rank_fuse as rf  # noqa: E402
import p02_rank_opening  # noqa: E402
import p02_rank_pseudo  # noqa: E402
from c10data import EXPECTED, LEVELS, players_for, read_split  # noqa: E402
from p02_rank_fuse import (BOOT_SEED, BOUNDS, FLOAT_EPS, GATE, N_BOOT, N_LEVELS, P02E9_MIN_ACC,  # noqa: E402,F401
                           PENALTY, PROB_COLUMNS, SELECT_TOL, STARTS, PlayerBootstrap, QSet, basis, build_qset,
                           check_val_logits, fit, load_dump, load_val_keys, log_softmax, metrics, params_dict,
                           pool_matrix, pooled, predict, question_logits, read_question_meta, rule_b_np, softmax)
from p02_rank_opening import load_opening  # noqa: E402
from p02_rank_pseudo import (FIT_MIN_GAMES, MIN_GAMES, P02Error, is_inside, load_inner_keys, load_npz,  # noqa: E402
                             load_pseudo, notebook_smoke_flag, now_iso, refuse_label_file, sha256_file, str_array,
                             write_json)

# ================================================================ constants
RESULTS = os.path.join(REPO, "results")
P02_RANK = os.path.join(RESULTS, "prereg02", "rank")
P03_ROOT = os.path.join(RESULTS, "prereg03")
P03_OUT = os.path.join(P03_ROOT, "rank")
SPLIT_CSV = os.path.join(RESULTS, "split_v1", "players_split.csv")
FALLBACK_R10 = os.path.join(RESULTS, "test02", "rank", "fallback_R10.json")
QUESTIONS = os.path.join(REPO, "data", "local", "rank_val_questions.csv")
READONLY_ROOTS = ("prereg01", "prereg02", "test02")   # under results/

T_STEPS = 33768                      # PREREG_03 1.1: 12 x 2,814, independent of workers and GPU
ARMS = ("a", "b")
N_SYMS = 8
POOL = "log"
C5_DUMP = "p02e9_d4"
SOURCES = {"C5": [(C5_DUMP, "all")], "C6A": [("p03a_d4", "all")], "C6B": [("p03b_d4", "all")]}
SOURCE_ARM = {"C6A": "a", "C6B": "b"}
ARM_SOURCE = {"a": "C6A", "b": "C6B"}
SOURCE_TEXT = {
    "C5": "prereg02 epoch 9, 8 symmetries (committed; R20's source)",
    "C6A": "prereg03 arm A final.pth (step T), 8 symmetries",
    "C6B": "prereg03 arm B final.pth (step T), 8 symmetries",
}
COMBOS = (("P00", ("C5",)), ("P01", ("C5", "C6A")), ("P02", ("C5", "C6B")), ("P03", ("C5", "C6A", "C6B")),
          ("P04", ("C6A",)), ("P05", ("C6B",)), ("P06", ("C6A", "C6B")))
WIRING_STOP_POINTS = 0.2             # swapped - correct accuracy above this -> bug
KAPPA_MIN_DACC_POINTS = 0.5          # the kappa requirement applies from this dacc on
VAL_LEVEL_QUESTIONS = dict(zip(LEVELS, (178, 163, 207, 308, 87, 122, 124, 128, 131, 141)))
R20_INNER_OOF = 0.5854702521055817   # results/prereg02/rank/tune_report.json R20 (registered)
R10_INNER_REGISTERED = 0.5687        # results/test02/rank/fallback_R10.json, rounded (PREREG_03 2.3)
R10_FORMAT = "c10-test02 rank fallback_R10 v1"
SELFCHECK_TOL = 1e-12
PARAM_KEYS = ("a", "gamma", "w", "beta", "b")
FROZEN_TOOL_FILES = ("p02_rank_fuse.py", "p02_rank_pseudo.py", "p02_rank_opening.py", "p02_moves.py",
                     "c10data.py", "decide.py")
OWN_FILE = "p03_rank_fuse.py"
FROZEN_FORMAT = "c10-prereg03 rank frozen_config v1"
CHECK_FORMAT = "c10-prereg03 rank check_report v1"
SELFCHECK_FORMAT = "c10-prereg03 rank selfcheck v1"
UNPACK_FORMAT = "c10-prereg03 unpack v1"
KAPPA_CHUNK = 20000
VAL_QUESTIONS_FULL, VAL_GAMES_FULL = 1589, 19156
ARM_RECORDS = ("config.json", "env.json", "timings.json", "run_log.json", "leak_check.json", "ckpt/history.csv",
               "readonly_before.json", "readonly_after.json")
ZIP_ALLOWED_EXT = (".json", ".csv", ".npz", ".txt", ".log")
# the notebook's stop_arm() statuses (registered stops; the arm is not a result). 'drive_space_pause' and a run that
# ended in a refusal before any computation (status None) are not stops: a later completed run in the folder is used
STOP_STATUSES = ("time_stop", "nan_stop", "pass_check_failed", "final_check_failed")
STOPPED_FILE = "STOPPED.json"
ZIP_MAX_BYTES = 100 * 1024 * 1024
# single-colour records (PREREG_03 rev 3; never used by the selection, the gates or the check-time colour stops;
# the kappa reference's test-day case "condition holds, validation sub-question dkappa not > 0" only withholds H3R,
# 11(c)): the official rank questions hold 5/10/15/20 games of one colour each
MONO_SALT = "c10-prereg03|mono"
MONO_KS = (5, 10, 15, 20)
MONO_ROUNDS = tuple(range(32))
MONO_COLOURS = ("B", "W")
MONO_FORMAT = "c10-prereg03 single-colour inner pseudo-questions v1"


def arm_stems(x):
    return ["inner_keys", "val_keys", f"inner_p03{x}_d4", f"inner_p03{x}_s0_swap", f"val_p03{x}_d4",
            f"val_p03{x}_s0_swap"]


def arm_required(x):
    """Files every arm result must hold (paths relative to rank_{x}/), plan.md 3.7."""
    return list(ARM_RECORDS) + [f"dumps/{s}.npz" for s in arm_stems(x)] + \
        [f"dumps/_DONE_{s}.json" for s in arm_stems(x)]


class P03Error(RuntimeError):
    pass


ERRORS = (P03Error, P02Error, decide.DecideError)


# ================================================================ small helpers
def _real(p):
    return os.path.normcase(os.path.realpath(p))


def same_path(a, b):
    return _real(a) == _real(b)


def read_json(path):
    refuse_label_file(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_bytes_text(b):
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def sha256_text(path):
    """SHA-256 of a source file with CRLF folded to LF (core.autocrlf=true may rewrite line ends)."""
    with open(path, "rb") as f:
        return sha256_bytes_text(f.read())


def environment():
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__}


def git(args, binary=False, repo=None):
    try:
        r = subprocess.run(["git", "-C", repo or REPO] + list(args), capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return None, (b"" if binary else str(e))
    return r.returncode, (r.stdout if binary else r.stdout.decode("utf-8", "replace").strip())


def repo_rel(path):
    return os.path.relpath(os.path.realpath(path), os.path.realpath(REPO)).replace(os.sep, "/")


def git_head():
    rc, head = git(["rev-parse", "HEAD"])
    return head if rc == 0 else None


def require_tracked_clean(paths, what):
    """Every path inside the repository, tracked by git and unchanged in the working tree."""
    outside = [p for p in paths if not is_inside(p, REPO)]
    if outside:
        raise P03Error(f"{what}: {outside[:2]} outside the repository (real mode runs on committed files)")
    rels = [repo_rel(p) for p in paths]
    if not rels:
        raise P03Error(f"{what}: no files")
    rc, _ = git(["ls-files", "--error-unmatch", "--"] + rels)
    if rc != 0:
        raise P03Error(f"{what}: not all committed ({rels[:3]}{' ...' if len(rels) > 3 else ''}); commit first")
    rc, st = git(["status", "--porcelain", "--"] + rels)
    if rc != 0 or st:
        raise P03Error(f"{what}: changed in the working tree: {st[:300]}")
    return {"files": len(rels), "git_head": git_head()}


def files_under(d):
    out = []
    for root, dirs, files in os.walk(d):
        dirs.sort()
        out.extend(os.path.join(root, f) for f in sorted(files))
    return out


def head_blob(rel):
    rc, b = git(["cat-file", "blob", f"HEAD:{rel}"], binary=True)
    return b if rc == 0 else None


def frozen_tools_check(tools_dir=None, blob=None):
    """The frozen p02 tools, c10data and decide equal their HEAD blobs (CRLF-insensitive); else refuse."""
    tools_dir = tools_dir or HERE
    blob = blob or head_blob
    rec, bad = {}, []
    for f in FROZEN_TOOL_FILES:
        path = os.path.join(tools_dir, f)
        work = sha256_text(path) if os.path.exists(path) else None
        b = blob("tools/" + f)
        head = sha256_bytes_text(b) if b is not None else None
        rec[f] = work
        if work is None or head is None or work != head:
            bad.append(f"{f} (working {str(work)[:12]}, HEAD {str(head)[:12]})")
    if bad:
        raise P03Error(f"frozen tools differ from HEAD: {bad}; the p02 tools, c10data.py and decide.py are never "
                       f"edited (PREREG_03 2.1)")
    return {"rule": "sha256 with CRLF folded to LF, working file == HEAD blob", "files_sha256": rec}


def imported_from_tools():
    mods = {"p02_rank_fuse.py": rf, "p02_rank_pseudo.py": p02_rank_pseudo, "p02_rank_opening.py": p02_rank_opening,
            "p02_moves.py": p02_moves, "c10data.py": c10data, "decide.py": decide}
    bad = [f for f, m in mods.items() if not same_path(m.__file__, os.path.join(HERE, f))]
    if bad:
        raise P03Error(f"modules imported from somewhere else than {HERE}: {bad}")


def tool_state():
    files = (OWN_FILE,) + FROZEN_TOOL_FILES
    rels = ["tools/" + f for f in files]
    # binary: git()'s text mode strips the output, which would cut the leading status blank of the first line
    rc, st = git(["status", "--porcelain", "--untracked-files=all", "--"] + rels, binary=True)
    lines = st.decode("utf-8", "replace").splitlines() if rc == 0 else None
    return {"git_head": git_head(), "dirty_or_untracked": ([ln[3:] for ln in lines if ln.strip()]
                                                           if lines is not None else None),
            "files_sha256_rule": "sha256 of the file with CRLF folded to LF",
            "files_sha256": {f: sha256_text(os.path.join(HERE, f)) for f in files}}


def require_tools_committed(tools):
    if tools["git_head"] is None or tools["dirty_or_untracked"] is None or tools["dirty_or_untracked"]:
        raise P03Error(f"commit the tools first (HEAD {tools['git_head']}, uncommitted "
                       f"{tools['dirty_or_untracked']})")


def check_smoke_size(n, full, what, smoke):
    if smoke and n >= full:
        raise P03Error(f"--smoke refused: {what} has {n} rows, the size of the real data ({full})")


# ================================================================ paths and modes
class Ctx:
    pass


def resolve(a, cmd):
    """Real mode: repository paths only (out-dir = results/prereg03/rank, or outside the repository for a
    check / selfcheck preview). --smoke: every input path given, SMOKE inputs, output outside the repository."""
    c = Ctx()
    c.smoke = bool(a.smoke)
    c.cmd = cmd
    if getattr(a, "min_acc", None) is not None and not c.smoke:
        raise P03Error("--min-acc may only change with --smoke (the floor is 24.15 %)")
    c.min_acc = P02E9_MIN_ACC if getattr(a, "min_acc", None) is None else float(a.min_acc)
    c.record_colour = bool(getattr(a, "smoke_record_colour", False))
    if c.record_colour and not c.smoke:
        raise P03Error("--smoke-record-colour is for SMOKE runs only (the colour stops always apply in real mode)")
    given = {"--p02-dir": a.p02_dir, "--p02-local-dir": a.p02_local_dir, "--p03-dir": a.p03_dir,
             "--split": a.split, "--fallback-r10": a.fallback_r10, "--questions": a.questions}
    real = {"--p02-dir": P02_RANK, "--p02-local-dir": P02_RANK, "--p03-dir": P03_ROOT, "--split": SPLIT_CSV,
            "--fallback-r10": FALLBACK_R10, "--questions": QUESTIONS}
    if c.smoke:
        need = ["--p02-dir", "--p02-local-dir", "--p03-dir", "--split"] + (["--questions"] if cmd == "apply" else [])
        missing = [k for k in need if not given[k]] + ([] if a.out_dir else ["--out-dir"])
        if missing:
            raise P03Error(f"--smoke needs {missing} (smoke never falls back to the repository paths)")
        vals = {k: (os.path.abspath(v) if v else None) for k, v in given.items()}
        c.out_dir = os.path.abspath(a.out_dir)
        if is_inside(c.out_dir, REPO):
            raise P03Error(f"--smoke output must be outside the repository (got {c.out_dir})")
        for k in ("--p02-dir", "--p02-local-dir", "--p03-dir"):
            if is_inside(vals[k], REPO):
                raise P03Error(f"--smoke {k} must be outside the repository (SMOKE dumps only; got {vals[k]})")
    else:
        for k, v in given.items():
            if v and not same_path(v, real[k]):
                raise P03Error(f"{k}: real mode uses {real[k]} only (use --smoke for other data)")
        vals = dict(real)
        c.out_dir = os.path.abspath(a.out_dir) if a.out_dir else P03_OUT
        if is_inside(c.out_dir, REPO) and not same_path(c.out_dir, P03_OUT):
            raise P03Error(f"--out-dir inside the repository must be {P03_OUT} (got {c.out_dir})")
    c.p02_dir, c.p02_local, c.p03_dir = vals["--p02-dir"], vals["--p02-local-dir"], vals["--p03-dir"]
    c.split, c.fallback_r10, c.questions = vals["--split"], vals["--fallback-r10"], vals["--questions"]
    c.p02_dumps = os.path.join(c.p02_dir, "dumps")
    c.arm_dir = {x: os.path.join(c.p03_dir, f"rank_{x}") for x in ARMS}
    for ro in READONLY_ROOTS:
        if is_inside(c.out_dir, os.path.join(RESULTS, ro)):
            raise P03Error(f"refusing to write into {c.out_dir}: results/{ro} is read-only for prereg03")
    for d in (c.p02_dir, c.p02_local) + tuple(c.arm_dir.values()):
        if is_inside(c.out_dir, d) or same_path(c.out_dir, d):
            raise P03Error(f"refusing to write into the input folder {d}")
    c.in_repo = is_inside(c.out_dir, REPO)
    c.preview = (not c.smoke) and not c.in_repo
    flag = notebook_smoke_flag(c.p02_dir)
    if c.smoke and flag is not True:
        raise P03Error(f"--smoke needs prereg02 SMOKE dumps: {c.p02_dir}/config.json must say \"smoke\": true "
                       f"(got {flag!r})")
    if not c.smoke and flag is not False:
        raise P03Error(f"{c.p02_dir}/config.json must say \"smoke\": false in real mode (got {flag!r})")
    return c


def start(c):
    """Common prologue: frozen tool hashes (HEAD, and in real mode prereg02's record)."""
    imported_from_tools()
    ft = frozen_tools_check()
    if not c.smoke:
        fc = read_json(os.path.join(P02_RANK, "frozen_config.json"))
        rec = (fc.get("tool") or {}).get("files_sha256") or {}
        diff = [f for f in FROZEN_TOOL_FILES if rec.get(f) != ft["files_sha256"][f]]
        if diff:
            raise P03Error(f"frozen tools differ from the hashes prereg02's tune recorded: {diff}")
        ft["equal_to_prereg02_frozen_config_record"] = True
    return ft


def p02_inputs(c, val=False):
    """prereg02's committed inputs (never recomputed). Real mode: SHA-256 equal to prereg02's records
    (frozen_config.json inputs for the inner files, apply_report.json for the validation files) and tracked."""
    files = {"inner_keys.npz": os.path.join(c.p02_dumps, "inner_keys.npz"),
             "pseudo_inner.json": os.path.join(c.p02_local, "pseudo_inner.json"),
             "opening_inner.npz": os.path.join(c.p02_local, "opening_inner.npz"),
             "players_split.csv": c.split,
             "inner_p02e9_d4.npz": os.path.join(c.p02_dumps, "inner_p02e9_d4.npz")}
    vfiles = {"val_p02e9_d4.npz": os.path.join(c.p02_dumps, "val_p02e9_d4.npz"),
              "val_keys.npz": os.path.join(c.p02_dumps, "val_keys.npz"),
              "opening_val.npz": os.path.join(c.p02_local, "opening_val.npz")}
    if val:
        files.update(vfiles)
    missing = [p for p in files.values() if not os.path.exists(p)]
    if missing:
        raise P03Error(f"prereg02 inputs missing: {missing}")
    sha = {k: sha256_file(p) for k, p in files.items()}
    rec = {"sha256": sha}
    if not c.smoke:
        fc = read_json(os.path.join(P02_RANK, "frozen_config.json"))
        ap = read_json(os.path.join(P02_RANK, "apply_report.json"))
        want = dict(fc["inputs_sha256"])
        want.update({k: ap["inputs_sha256"][k] for k in vfiles})
        bad = [k for k in sha if want.get(k) != sha[k]]
        if bad:
            raise P03Error(f"prereg02 inputs differ from prereg02's records: {bad}")
        rec["equal_to_prereg02_records"] = True
        rec["commit_check"] = require_tracked_clean(list(files.values()), "prereg02 inputs")
    return rec


# ================================================================ sources and combinations
def combo_sources(cid):
    return dict(COMBOS)[cid]


def combo_dumps(sources):
    return [n for s in sources for n, _ in SOURCES[s]]


def combo_logprobs(sources, logits):
    """Per-game log-probabilities [N,10] of a combination: the orientation lists of its sources concatenated,
    log_softmax(mean of the log_softmax arrays) as p02's source_logprobs (every source has 8 orientations, so
    every source has the same weight; one source alone is exactly p02's source of that dump)."""
    parts = []
    for s in sources:
        for name, which in SOURCES[s]:
            z = logits[name]
            if which != "all" or z.ndim != 3 or z.shape[1] != N_SYMS:
                raise P03Error(f"{s}: {name} must have {N_SYMS} orientations (got {z.shape}); equal weights need it")
            z = z.astype(np.float64)
            parts.extend(log_softmax(z[:, k, :]) for k in range(z.shape[1]))
    if len(parts) == 1:
        return parts[0]
    return log_softmax(np.mean(parts, axis=0))


def available_combos(usable):
    """[(id, sources, status, reason)] in the fixed order; a combination needs every source usable."""
    out = []
    for cid, srcs in COMBOS:
        bad = [s for s in srcs if s not in usable]
        out.append((cid, srcs, "ok" if not bad else "dropped",
                    None if not bad else f"unusable or missing source(s) {bad}"))
    return out


def select(results):
    """results: [{"id", "status", "oof": {"rank_balanced"}}] in the fixed order -> (chosen, record)."""
    ok = [r for r in results if r["status"] == "ok"]
    best = max(r["oof"]["rank_balanced"] for r in ok)
    within = [r["id"] for r in ok if best - r["oof"]["rank_balanced"] <= SELECT_TOL + FLOAT_EPS]
    chosen = next(r for r in ok if r["id"] == within[0])
    return chosen, {"rule": f"highest OOF rank-balanced mean (rule B); within {SELECT_TOL} of the best -> the "
                            f"earliest in the list P00..P06 (compared with {FLOAT_EPS} float slack)",
                    "best_id": next(r["id"] for r in ok if r["oof"]["rank_balanced"] == best),
                    "best_rank_balanced": best, "selected": chosen["id"],
                    "selected_rank_balanced": chosen["oof"]["rank_balanced"], "within_tolerance": within}


def gate_record(sel_id, rb_sel, rb_p00, sc_sel, sc_p00, boot):
    """G1-G3 of PREREG_03 2.4 for the selected combination (P00 itself -> not submitted)."""
    if sel_id == "P00":
        return {"selected": "P00", "G1": None, "G2": None, "G3": None, "passed": False,
                "reason": "P00 (R20) selected: not submitted, R20 unchanged, the scoring slot is kept"}
    delta = rb_sel - rb_p00
    d = boot.stats(sc_sel) - boot.stats(sc_p00)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    g3 = float(np.mean(np.asarray(sc_sel) - np.asarray(sc_p00)))
    g1 = {"threshold": GATE, "delta": delta, "passed": bool(delta >= GATE - FLOAT_EPS)}
    g2 = {"ci95": [lo, hi], "se": float(np.std(d, ddof=1)), "passed": bool(lo > 0),
          "bootstrap": f"players within each level, {N_BOOT} draws, default_rng({BOOT_SEED}), paired"}
    g3r = {"question_mean_difference": g3, "passed": bool(g3 >= -FLOAT_EPS),
           "note": "mean of the per-question score differences on the inner eval rounds, >= 0 with 1e-12 slack"}
    passed = g1["passed"] and g2["passed"] and g3r["passed"]
    return {"selected": sel_id, "G1": g1, "G2": g2, "G3": g3r, "passed": bool(passed),
            "reason": "all three gates passed" if passed else "a gate failed: not submitted, R20 unchanged"}


def se_val_record(delta, d, y):
    """SE_val = sqrt(sum_l Var_l(d) / N_l) / 10 and Phi(delta / SE_val - 1.96) (recorded only)."""
    d = np.asarray(d, dtype=np.float64)
    terms, var = {}, 0.0
    for i, lv in enumerate(LEVELS):
        v = d[np.asarray(y) == i]
        if len(v) >= 2:
            t = float(np.var(v, ddof=1)) / VAL_LEVEL_QUESTIONS[lv]
            terms[lv] = t
            var += t
    se = math.sqrt(var) / 10.0
    prob = None if se <= 0 else 0.5 * (1.0 + math.erf((delta / se - 1.96) / math.sqrt(2.0)))
    return {"se_val": se, "adoption_probability": prob, "delta": delta,
            "N_l": VAL_LEVEL_QUESTIONS, "per_level_var_over_N": terms,
            "formula": "sqrt(sum_l Var_l(d, ddof 1) / N_l) / 10; Phi(delta / SE_val - 1.96)"}


# ================================================================ colour checks
def kappa_questions(lp, groups):
    """Per-question kappa: mean over the question's games g of sum_k softmax(mean_{h != g} l_h)_k * l_g,k
    (lp [N,10] per-game log-probabilities, groups = row lists); NaN for questions with < 2 games."""
    lp = np.asarray(lp, dtype=np.float64)
    sizes = np.array([len(g) for g in groups], dtype=np.int64)
    out = np.full(len(groups), np.nan)
    idx = np.nonzero(sizes >= 2)[0]
    for s in range(0, len(idx), KAPPA_CHUNK):
        qs = idx[s:s + KAPPA_CHUNK]
        n = sizes[qs]
        rows = np.concatenate([np.asarray(groups[q], dtype=np.int64) for q in qs])
        qi = np.repeat(np.arange(len(qs)), n)
        L = lp[rows]
        tot = np.add.reduceat(L, np.concatenate([[0], np.cumsum(n)[:-1]]), axis=0)   # per-question sum of l
        others = (tot[qi] - L) / (n[qi] - 1)[:, None]
        ell = np.sum(softmax(others) * L, axis=1)
        out[qs] = np.bincount(qi, weights=ell, minlength=len(qs)) / n
    return out


def boot_plain_mean(boot, values):
    """PlayerBootstrap draws (players resampled within each level) of the plain question mean."""
    s = np.bincount(boot.player, weights=values, minlength=boot.n_players)
    num = sum(s[d].sum(1) for d in boot.draws)
    den = sum(boot.count[d].sum(1) for d in boot.draws)
    return num / den


def colour_decision(acc_correct, acc_swap, dk_inner_lo, dk_val):
    """Items 3-4 of PREREG_03 2.2 -> (stops, record)."""
    dacc = acc_correct - acc_swap
    stops = []
    wiring = (acc_swap - acc_correct) > WIRING_STOP_POINTS
    if wiring:
        stops.append(f"colour wiring: swapped accuracy {acc_swap:.4f}% exceeds the correct {acc_correct:.4f}% by "
                     f"more than {WIRING_STOP_POINTS} points (bug; revise before re-exporting or retraining)")
    applies = bool(dacc >= KAPPA_MIN_DACC_POINTS and dk_inner_lo is not None and dk_inner_lo > 0)
    kappa_ok = None
    if applies:
        kappa_ok = bool(dk_val is not None and dk_val > 0)
        if not kappa_ok:
            stops.append(f"colour semantics: the model uses the colour on inner (dacc {dacc:.3f} points, dkappa CI "
                         f"lower {dk_inner_lo:.3g} > 0) but the validation dkappa is {dk_val} (not > 0): stop and "
                         f"find the bug")
    return stops, {"dacc_points": dacc, "wiring_stop": bool(wiring), "kappa_requirement_applies": applies,
                   "kappa_val_positive": kappa_ok,
                   "rule": f"stop when swapped - correct > {WIRING_STOP_POINTS} points; when dacc >= "
                           f"{KAPPA_MIN_DACC_POINTS} points and the inner dkappa CI lower bound > 0 the validation "
                           f"dkappa must be > 0"}


# ================================================================ single-colour records (PREREG_03 rev 3, record only)
def mono_order_key(rank, game_id):
    return hashlib.sha256(f"{MONO_SALT}|order|{rank}|{game_id}".encode("utf-8")).hexdigest()


def mono_draw(r, player_id, colour, k, n):
    """Positions (in the ordered pool of n games) of round r's k-game question of (player, colour)."""
    seed = int(hashlib.sha256(f"{MONO_SALT}|q|{r}|{player_id}|{colour}|{k}".encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed).sample(range(n), k)


def mono_questions(keys, players):
    """Single-colour inner pseudo-questions -> [(round, player index, colour, k, [inner dump rows])].

    players: pseudo_inner.json's bank players (their order and folds). Pool of (player, colour) = every dumped inner
    game of that player with that colour, ordered by sha256('c10-prereg03|mono|order|{rank}|{game_id}') (hex). Round r
    (MONO_ROUNDS), every player in player order, colour B then W, k in MONO_KS with k <= pool size: positions
    random.Random(int(sha256('c10-prereg03|mono|q|{r}|{player_id}|{colour}|{k}')[:16], 16)).sample(range(pool), k).
    One generator per question, so a question never depends on which other players or colours are present."""
    index = {p["player_id"]: i for i, p in enumerate(players)}
    pools = {}
    for i, (pid, col) in enumerate(zip(keys["player_id"], keys["color"])):
        pi = index.get(pid)
        if pi is None:
            continue
        if keys["rank"][i] != players[pi]["rank"]:
            raise P03Error(f"inner game {keys['game_id'][i]}: level {keys['rank'][i]} differs from its player's")
        pools.setdefault((pi, col), []).append(i)
    ordered = []
    for pi in range(len(players)):
        for col in MONO_COLOURS:
            rows = sorted(pools.get((pi, col), []), key=lambda i: mono_order_key(keys["rank"][i], keys["game_id"][i]))
            ordered.append((pi, col, rows))
    out = []
    for r in MONO_ROUNDS:
        for pi, col, rows in ordered:
            for k in MONO_KS:
                if len(rows) >= k:
                    pos = mono_draw(r, players[pi]["player_id"], col, k, len(rows))
                    out.append((r, pi, col, k, [rows[j] for j in pos]))
    return out


def mono_set(inner):
    """The single-colour inner pseudo-questions as arrays, with their SHA-256 and counts (inner files only)."""
    keys, players = inner["keys"], inner["players"]
    qs = mono_questions(keys, players)
    h = hashlib.sha256()
    for r, pi, col, k, rows in qs:
        h.update(f"{r}|{players[pi]['player_id']}|{col}|{k}|{' '.join(keys['game_id'][i] for i in rows)}\n"
                 .encode("utf-8"))
    player = np.array([q[1] for q in qs], dtype=np.int64)
    y = np.array([LEVELS.index(players[q[1]]["rank"]) for q in qs], dtype=np.int64)
    colour = np.array([q[2] for q in qs], dtype="<U1")
    ks = np.array([q[3] for q in qs], dtype=np.int64)
    first = [q for q in qs if q[0] == MONO_ROUNDS[0]] if MONO_ROUNDS else []
    pairs = {lv: {c: 0 for c in MONO_COLOURS} for lv in LEVELS}
    for _r, pi, col, k, _rows in first:
        if k == MONO_KS[0]:
            pairs[players[pi]["rank"]][col] += 1
    counts = {"questions": len(qs), "players": int(len(np.unique(player))),
              "player_colour_pairs": sum(sum(v.values()) for v in pairs.values()),
              "player_colour_pairs_by_level": pairs,
              "questions_by_colour": {c: int((colour == c).sum()) for c in MONO_COLOURS},
              "questions_by_k": {str(k): int((ks == k).sum()) for k in MONO_KS},
              "games_per_question_mean": float(np.mean([len(q[4]) for q in qs])) if qs else None}
    definition = {
        "format": MONO_FORMAT,
        "players": "pseudo_inner.json's bank players (split v1 inner, >= 5 inner games) with their folds",
        "pool": "(player, colour): every dumped inner game of that player with that colour (inner_keys.npz colour), "
                "ordered by sha256('c10-prereg03|mono|order|{rank}|{game_id}') hex",
        "questions": f"rounds {list(MONO_ROUNDS[:1]) + list(MONO_ROUNDS[-1:])} ({len(MONO_ROUNDS)} rounds); every "
                     f"player in player order, colour B then W, k in {list(MONO_KS)} with k <= pool size: "
                     f"random.Random(int(sha256('c10-prereg03|mono|q|{{r}}|{{player_id}}|{{colour}}|{{k}}')[:16], "
                     f"16)).sample(range(pool), k)",
        "why": "official competition 2 questions hold 5/10/15/20 games of one colour each (AIdea data page, "
               "docs/DATA.md); the registered inner pseudo-questions and the local validation are mixed-colour"}
    return {"questions": qs, "groups": [q[4] for q in qs], "player": player, "y": y,
            "fold": np.array([players[q[1]]["fold"] for q in qs], dtype=np.int64), "colour": colour, "k": ks,
            "n_players": len(players), "sha256": h.hexdigest(), "counts": counts, "definition": definition}


def mono_qset(inner, ms):
    """QSet (games per question, level, player, fold, opening mean) and pooling matrix of the single-colour
    questions, built as p02's build_qset builds the eval questions."""
    n_rows = inner["keys"]["n"]
    S = pool_matrix(ms["groups"], n_rows)
    Sop = pool_matrix([[r for r in g if inner["op_has"][r]] for g in ms["groups"]], n_rows)
    X = QSet([len(g) for g in ms["groups"]], ms["y"], ms["player"], ms["fold"], np.asarray(Sop @ inner["op_o"]))
    return X, S


def score_summary(sc, y, mask=None):
    """Rank-balanced mean (mean over the levels present of the per-level question mean) and question mean."""
    sc, y = np.asarray(sc, dtype=np.float64), np.asarray(y)
    if mask is not None:
        sc, y = sc[mask], y[mask]
    per = {lv: float(sc[y == i].mean()) for i, lv in enumerate(LEVELS) if (y == i).any()}
    return {"rank_balanced": float(np.mean(list(per.values()))) if per else None,
            "question_mean": float(sc.mean()) if len(sc) else None, "n_questions": int(len(sc)),
            "levels_present": len(per), "per_level": per}


def sub_bootstrap(ms, mask):
    """PlayerBootstrap (players within each level, N_BOOT, default_rng(BOOT_SEED)) of a subset of the questions."""
    return PlayerBootstrap(ms["player"][mask], ms["y"][mask], ms["n_players"])


def ci95(draws):
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def mono_score_record(pred, X, ms, boot):
    """p02's metrics on the single-colour questions plus the colour and k splits (record only)."""
    m, sc = metrics(pred, X.y, X.n, X.player)
    m.update(boot.summary(sc))
    m["by_colour"] = {c: score_summary(sc, X.y, ms["colour"] == c) for c in MONO_COLOURS}
    m["by_k"] = {str(k): score_summary(sc, X.y, ms["k"] == k) for k in MONO_KS}
    return m, sc


def mono_diff_record(ms, boot, sc_a, sc_b):
    """Paired PlayerBootstrap difference a - b on the single-colour questions, all and per colour (record only)."""
    def one(mask, bt):
        a, b = score_summary(sc_a, ms["y"], mask), score_summary(sc_b, ms["y"], mask)
        d = bt.stats(sc_a if mask is None else sc_a[mask]) - bt.stats(sc_b if mask is None else sc_b[mask])
        return {"rank_balanced": a["rank_balanced"] - b["rank_balanced"],
                "question_mean": a["question_mean"] - b["question_mean"], "ci95": ci95(d),
                "se": float(np.std(d, ddof=1)), "n_questions": a["n_questions"]}
    out = one(None, boot)
    out["by_colour"] = {c: (one(ms["colour"] == c, sub_bootstrap(ms, ms["colour"] == c))
                            if (ms["colour"] == c).any() else None) for c in MONO_COLOURS}
    out["by_k"] = {str(k): ({"rank_balanced": score_summary(sc_a, ms["y"], ms["k"] == k)["rank_balanced"] -
                             score_summary(sc_b, ms["y"], ms["k"] == k)["rank_balanced"]}
                            if (ms["k"] == k).any() else None) for k in MONO_KS}
    out["bootstrap"] = (f"paired, players within each level, {N_BOOT} draws, default_rng({BOOT_SEED}); the colour "
                        f"splits draw their own players")
    return out


def val_colour_subquestions(vk):
    """Each validation question's parsed games split by the colour column of the question file (val_keys colour;
    no label) -> [(question index, colour, rows)], sub-questions with >= 2 parsed games."""
    out = []
    for qi, (_q, rows) in enumerate(vk["groups"]):
        for col in MONO_COLOURS:
            g = [i for i in rows if vk["parsed"][i] and vk["color"][i] == col]
            if len(g) >= 2:
                out.append((qi, col, g))
    return out


def question_boot_mean(qidx, values):
    """Question-level bootstrap (no player id, no label): resample the questions (N_BOOT, default_rng(BOOT_SEED));
    statistic = the plain mean over the drawn questions' sub-questions."""
    qidx = np.asarray(qidx)
    uq, inv = np.unique(qidx, return_inverse=True)
    s = np.bincount(inv, weights=np.asarray(values, dtype=np.float64), minlength=len(uq))
    cnt = np.bincount(inv, minlength=len(uq)).astype(np.float64)
    d = np.random.default_rng(BOOT_SEED).integers(0, len(uq), size=(N_BOOT, len(uq)))
    return s[d].sum(1) / cnt[d].sum(1)


def kappa_single_colour(lp_c, lp_s, ms, boot, dacc, vk, vlp_c, vlp_s):
    """Single-colour kappa reference (record only; PREREG_03 rev 3): dkappa = kappa(correct) - kappa(whole question
    colour swapped) on the single-colour inner pseudo-questions (PlayerBootstrap CI) and on the validation questions'
    single-colour sub-questions (question file colours, no label; question-level bootstrap CI recorded)."""
    rec = {"status": "recorded only (PREREG_03 rev 3): reference for the test-day kappa check on the official "
                     "single-colour questions; the check-time rule (item 4, mixed-colour) is unchanged"}
    if not ms["questions"]:
        rec["inner"] = None
    else:
        kc, ks = kappa_questions(lp_c, ms["groups"]), kappa_questions(lp_s, ms["groups"])
        dk = kc - ks
        inner = {"correct": float(np.mean(kc)), "swapped": float(np.mean(ks)), "dkappa": float(np.mean(dk)),
                 "ci95": ci95(boot_plain_mean(boot, dk)), "questions": int(len(dk)),
                 "players": ms["counts"]["players"], "questions_sha256": ms["sha256"],
                 "bootstrap": f"PlayerBootstrap draws (players within each level, {N_BOOT}, default_rng({BOOT_SEED})) "
                              f"of the plain question mean; the colour splits draw their own players",
                 "by_colour": {}, "by_k": {}}
        for col in MONO_COLOURS:
            mk = ms["colour"] == col
            inner["by_colour"][col] = ({"dkappa": float(np.mean(dk[mk])), "correct": float(np.mean(kc[mk])),
                                        "swapped": float(np.mean(ks[mk])),
                                        "ci95": ci95(boot_plain_mean(sub_bootstrap(ms, mk), dk[mk])),
                                        "questions": int(mk.sum())} if mk.any() else None)
        for k in MONO_KS:
            mk = ms["k"] == k
            inner["by_k"][str(k)] = {"dkappa": float(np.mean(dk[mk])), "questions": int(mk.sum())} if mk.any() else None
        rec["inner"] = inner
    sub = val_colour_subquestions(vk)
    if sub:
        groups = [g for _, _, g in sub]
        qidx = np.array([q for q, _, _ in sub])
        cols = np.array([c for _, c, _ in sub])
        kvc, kvs = kappa_questions(vlp_c, groups), kappa_questions(vlp_s, groups)
        dkv = kvc - kvs
        val = {"correct": float(np.mean(kvc)), "swapped": float(np.mean(kvs)), "dkappa": float(np.mean(dkv)),
               "sub_questions": int(len(sub)), "questions_with_a_sub_question": int(len(np.unique(qidx))),
               "question_bootstrap_ci95_recorded_only": ci95(question_boot_mean(qidx, dkv)),
               "definition": "each validation question's parsed games split by their colour (color_i of the "
                             "question file, via val_keys.npz; no label); sub-questions with >= 2 parsed games; "
                             "plain mean over sub-questions",
               "by_colour": {}}
        for col in MONO_COLOURS:
            mk = cols == col
            val["by_colour"][col] = ({"dkappa": float(np.mean(dkv[mk])), "sub_questions": int(mk.sum()),
                                      "question_bootstrap_ci95_recorded_only": ci95(question_boot_mean(qidx[mk],
                                                                                                       dkv[mk]))}
                                     if mk.any() else None)
    else:
        val = None
    rec["val_sub_questions"] = val
    lo = rec["inner"]["ci95"][0] if rec["inner"] else None
    cond = bool(dacc >= KAPPA_MIN_DACC_POINTS and lo is not None and lo > 0)
    rec["test_day_reference"] = {
        "single_colour_condition": cond, "dacc_points": dacc, "inner_dkappa_ci95_lower": lo,
        "val_sub_question_dkappa": val["dkappa"] if val else None,
        "val_sub_question_dkappa_positive": (bool(val["dkappa"] > 0) if val else None),
        "rule": f"condition = dacc >= {KAPPA_MIN_DACC_POINTS} points and the single-colour inner dkappa CI lower "
                f"bound > 0; recorded here, applied as PREREG_03 rev 3 11(c) says (test day; condition true with "
                f"the validation sub-question dkappa not > 0 also withholds H3R at tune)"}
    return rec


# ================================================================ inner preparation (p02 tune's, verbatim steps)
def prepare_inner(c):
    ddir = c.p02_dumps
    keys = load_inner_keys(ddir)
    if c.smoke:
        check_smoke_size(keys["n"], EXPECTED["inner"]["games"], "inner_keys.npz", True)
    elif keys["n"] != EXPECTED["inner"]["games"]:
        raise P03Error(f"inner_keys has {keys['n']} games, expected {EXPECTED['inner']['games']}")
    pseudo = load_pseudo(os.path.join(c.p02_local, "pseudo_inner.json"), verify=True)
    pobj = pseudo["obj"]
    if pobj["inputs"]["inner_keys_sha256"] != keys["sha256"]:
        raise P03Error("pseudo_inner.json was built from a different inner_keys.npz")
    if not c.smoke and (pobj["smoke"] or pobj["params"]["min_games"] != MIN_GAMES
                        or pobj["params"]["fit_min_games"] != FIT_MIN_GAMES):
        raise P03Error("pseudo_inner.json is a smoke file or has non-registered thresholds")
    split = read_split(c.split)
    inner_set = players_for(split, "inner")
    players = pseudo["players"]
    outside = sorted({p for p in keys["player_id"] if p not in inner_set}
                     | {p["player_id"] for p in players if p["player_id"] not in inner_set})
    if outside:
        raise P03Error(f"{len(outside)} player_ids are not split-v1 inner players, e.g. {outside[:3]}")
    assert all(p in inner_set for p in keys["player_id"])  # every player_id is an inner player
    dump_row = {(r, g): i for i, (r, g) in enumerate(zip(keys["rank"], keys["game_id"]))}
    player_bank = []
    for p in players:
        rows = []
        for gid in p["bank"]:
            i = dump_row.get((p["rank"], gid))
            if i is None or keys["player_id"][i] != p["player_id"]:
                raise P03Error(f"bank game {gid} of {p['player_id']} is not in inner_keys for that player")
            rows.append((p["rank"], gid))
        player_bank.append(rows)
    fold_of = [p["fold"] for p in players]
    op = load_opening(os.path.join(c.p02_local, "opening_inner.npz"), "inner")
    op_row = {(r, g): i for i, (r, g) in enumerate(zip(str_array(op["rank"]), str_array(op["game_id"])))}
    op_pid = str_array(op["player_id"])
    idx = []
    for i, (r, g) in enumerate(zip(keys["rank"], keys["game_id"])):
        j = op_row.get((r, g))
        if j is None or op_pid[j] != keys["player_id"][i]:
            raise P03Error(f"inner game ({r}, {g}) missing from opening_inner.npz or player differs")
        idx.append(j)
    idx = np.asarray(idx)
    op_o = np.asarray(op["o"], dtype=np.float64)[idx]
    op_has = np.asarray(op["has"], dtype=bool)[idx]
    X_fit, S_fit = build_qset(pseudo["fit"], players, dump_row, fold_of, keys["n"], op_o, op_has, player_bank)
    X_eval, S_eval = build_qset(pseudo["eval"], players, dump_row, fold_of, keys["n"], op_o, op_has, player_bank)
    eval_groups = [[dump_row[player_bank[pi][k]] for k in pos] for _, pi, pos in pseudo["eval"]]
    assert isinstance(X_fit, QSet) and isinstance(X_eval, QSet)
    return {"keys": keys, "pseudo": pseudo, "players": players, "X_fit": X_fit, "S_fit": S_fit,
            "X_eval": X_eval, "S_eval": S_eval, "eval_groups": eval_groups, "op": op, "op_o": op_o, "op_has": op_has,
            "boot": PlayerBootstrap(X_eval.player, X_eval.y, len(players)),
            "inputs": {"inner_keys.npz": keys["sha256"], "pseudo_inner.json": pseudo["sha256"],
                       "opening_inner.npz": op["sha256"], "players_split.csv": sha256_file(c.split)}}


def load_c5_inner(c, n):
    d = load_dump(os.path.join(c.p02_dumps, f"inner_{C5_DUMP}.npz"), n, N_SYMS)
    if not np.isfinite(d["logits"]).all():
        raise P03Error(f"{d['path']}: non-finite logits")
    return d


def add_combo(inner, key, lp):
    for X, S in ((inner["X_fit"], inner["S_fit"]), (inner["X_eval"], inner["S_eval"])):
        X.m[(key, POOL)] = pooled(S, lp, POOL)


def run_oof(inner, key, opening, smoke, B):
    """p02 tune's 5-fold OOF loop: fold f fits on fit questions of the other folds, predicts eval fold f."""
    X_fit, X_eval = inner["X_fit"], inner["X_eval"]
    oof_pred = np.full(len(X_eval), -1, dtype=np.int64)
    oof_P = np.full((len(X_eval), N_LEVELS), np.nan)
    folds = []
    for f in range(5):
        ev = X_eval.fold == f
        tr = X_fit.fold != f
        if not tr.any():
            if not smoke:
                raise P03Error(f"fold {f}: no fit questions")
            theta = rf.uncalibrated_theta(opening)
            frec = {"fit_questions": 0, "fit_empty_smoke": True, "params": params_dict(theta, opening, B)}
        else:
            theta, frec = fit(X_fit.subset(tr), key, POOL, opening, B)
            frec["fit_questions"] = int(tr.sum())
        if ev.any():
            P = predict(theta, X_eval.subset(ev), key, POOL, opening, B)
            oof_P[ev] = P
            oof_pred[ev] = rule_b_np(P)
        frec["fold"] = f
        frec["eval_questions"] = int(ev.sum())
        folds.append(frec)
    assert (oof_pred >= 0).all()
    m, sc = metrics(oof_pred, X_eval.y, X_eval.n, X_eval.player)
    return m, sc, folds, oof_P


def fold_predict(X, key, folds, opening, B):
    """Probabilities of the questions of X with each fold's OOF parameters (the parameters run_oof fitted on the
    other folds' fit questions, read back from its fold records): row q uses the fold of its player."""
    P = np.full((len(X), N_LEVELS), np.nan)
    for frec in folds:
        ev = X.fold == frec["fold"]
        if ev.any():
            P[ev] = predict(theta_of(frec["params"], opening), X.subset(ev), key, POOL, opening, B)
    return P


def refit_all(inner, key, opening, smoke, B):
    X = inner["X_fit"]
    if len(X) == 0:
        if not smoke:
            raise P03Error("no fit questions for the refit")
        theta = rf.uncalibrated_theta(opening)
        return theta, {"fit_questions": 0, "fit_empty_smoke": True}
    theta, rec = fit(X, key, POOL, opening, B)
    rec["fit_questions"] = len(X)
    return theta, rec


# ================================================================ validation side (selfcheck, apply)
def load_val_side(c, sources, arm_dirs):
    """val_keys (equal in every folder used), the sources' validation dumps (masked as p02 apply) and the
    opening scores aligned to the slots."""
    vk = load_val_keys(c.p02_dumps)
    check_smoke_size(vk["n"], VAL_GAMES_FULL, "val_keys.npz", c.smoke)
    logits, sha, wsha = {}, {}, {}
    for s in sources:
        for name, _ in SOURCES[s]:
            ddir = c.p02_dumps if s == "C5" else os.path.join(arm_dirs[SOURCE_ARM[s]], "dumps")
            if s != "C5":
                diff = key_array_diff(os.path.join(ddir, "val_keys.npz"), os.path.join(c.p02_dumps, "val_keys.npz"),
                                      ("question_id", "slot", "color", "parsed"))
                if diff:
                    raise P03Error(f"{ddir}/val_keys.npz differs from prereg02's: {diff}")
            d = load_dump(os.path.join(ddir, f"val_{name}.npz"), vk["n"], N_SYMS)
            check_val_logits(d, vk["parsed"])
            logits[name] = np.where(vk["parsed"][:, None, None], d["logits"], 0.0).astype(np.float32)
            sha[f"val_{name}.npz"] = d["sha256"]
            wsha[name] = d["weights_sha256"]
    return vk, logits, sha, wsha


def opening_val_aligned(c, vk, build_id):
    op = load_opening(os.path.join(c.p02_local, "opening_val.npz"), "val")
    if build_id is not None and op["meta"]["build_id"] != build_id:
        raise P03Error("opening_val.npz is from a different build than the frozen opening_inner.npz")
    pos = {(q, int(s)): i for i, (q, s) in enumerate(zip(str_array(op["question_id"]),
                                                       np.asarray(op["slot"]).tolist()))}
    oi = []
    for i, (q, s) in enumerate(zip(vk["question_id"], vk["slot"])):
        j = pos.get((q, s))
        if j is None or str(op["color"][j]) != vk["color"][i]:
            raise P03Error(f"opening_val.npz: no matching entry for question {q} slot {s}")
        oi.append(j)
    oi = np.asarray(oi)
    return np.asarray(op["o"], dtype=np.float64)[oi], np.asarray(op["has"], dtype=bool)[oi], op


def val_probs(sources, theta, opening, B, vk, logits, op_o, op_has):
    """p02 apply's computation for a combination: [Q,10] probabilities (NaN rows when n_parsed = 0)."""
    lp = combo_logprobs(sources, logits)
    groups = vk["groups"]
    cnn_groups = [[i for i in rows if vk["parsed"][i]] for _, rows in groups]
    S = pool_matrix(cnn_groups, vk["n"])
    m = pooled(S, lp, POOL)
    n = np.array([len(g) for g in cnn_groups], dtype=np.float64)
    if opening:
        mop = np.asarray(pool_matrix([[i for i in rows if op_has[i]] for _, rows in groups], vk["n"]) @ op_o)
    else:
        mop = np.zeros_like(m)
    ok = n > 0
    P = np.full((len(groups), N_LEVELS), np.nan)
    if ok.any():
        z, _ = question_logits(theta, m[ok], mop[ok], n[ok], opening, B)
        P[ok] = softmax(z)
    return P, n, ok


def theta_of(par, opening):
    return np.array([par["a"], par["gamma"]] + ([par["w"]] if opening else []) + list(par["beta"]),
                    dtype=np.float64)


def params_diff(p, q):
    if len(p["beta"]) != len(q["beta"]) or len(p["b"]) != len(q["b"]):
        return float("inf")
    vals = [abs(p[k] - q[k]) for k in ("a", "gamma", "w")]
    vals += [abs(x - y) for k in ("beta", "b") for x, y in zip(p[k], q[k])]
    return float(max(vals))


# ================================================================ arm structure
def key_array_diff(path, ref_path, names):
    """Array-by-array comparison of two key files: [] when equal (values and shapes; dtype recorded)."""
    a = load_npz(path, names)
    b = load_npz(ref_path, names)
    out = []
    if sorted(a) != sorted(b):
        out.append(f"array names {sorted(a)} vs {sorted(b)}")
    for k in names:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape or not np.array_equal(x, y):
            out.append(k)
    return out


def load_swap(path, n):
    z = load_npz(path, ["logits", "weights_sha256"])
    lg = np.asarray(z["logits"])
    if lg.dtype != np.float32:
        raise P03Error(f"{path}: logits must be float32, got {lg.dtype}")
    if lg.ndim == 3 and lg.shape[1] == 1:
        lg = lg[:, 0, :]
    if lg.shape != (n, N_LEVELS):
        raise P03Error(f"{path}: logits shape {lg.shape}, expected [{n}, 10]")
    if "syms" in z and int(z["syms"]) != 1:
        raise P03Error(f"{path}: syms {int(z['syms'])}, expected 1 (original orientation only)")
    return {"logits": lg, "weights_sha256": str(z["weights_sha256"]), "path": path, "sha256": sha256_file(path)}


def marker_check(ddir, stem, rows, weights_sha, syms):
    path = os.path.join(ddir, f"_DONE_{stem}.json")
    mk = read_json(path)
    problems = []
    fsha = sha256_file(os.path.join(ddir, f"{stem}.npz"))
    if mk.get("sha256") != fsha:
        problems.append(f"{stem}: marker sha256 differs from the file")
    if mk.get("rows") != rows:
        problems.append(f"{stem}: marker rows {mk.get('rows')} != {rows}")
    if weights_sha is not None and mk.get("weights_sha256") != weights_sha:
        problems.append(f"{stem}: marker weights_sha256 differs from the dump's")
    if syms is not None and mk.get("syms") not in (syms, None):
        problems.append(f"{stem}: marker syms {mk.get('syms')} != {syms}")
    w = mk.get("weights")
    if weights_sha is not None and w and os.path.basename(str(w)) != "final.pth":
        problems.append(f"{stem}: marker weights {w!r} is not final.pth")
    return problems, {k: mk.get(k) for k in ("rows", "syms", "sha256", "weights", "weights_sha256", "gpu_name",
                                             "torch", "cudnn", "run", "seconds")}


def config_t(cfg):
    for path in (("recipe", "T"), ("T",), ("recipe", "T_STEPS"), ("T_STEPS",), ("t_steps",)):
        v = cfg
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, int):
            return v, ".".join(path)
    return None, None


def config_arm(cfg):
    for path in (("recipe", "ARM"), ("ARM",), ("arm",), ("recipe", "arm")):
        v = cfg
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, str):
            return v
    return None


def read_history(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps = []
    for r in rows:
        try:
            steps.append(int(float(r.get("global_step", ""))))
        except ValueError:
            pass
    return rows, (max(steps) if steps else None)


def last_run_status(run_log):
    runs = (run_log or {}).get("runs") or []
    return (runs[-1] or {}).get("status") if runs else None


def stop_record_problems(run_log, cfg, names):
    """PREREG_03 1.3 'any stop: that arm is not a result': problems when the arm's Drive folder ever recorded one of the
    notebook's registered stops (stop_arm), even if a later run in the same folder completed (the precedent is test02's
    FAILED_RUN_STATUSES, PREREG_02 rev 7). names: the arm's files (relative paths)."""
    out = []
    runs = (run_log or {}).get("runs") if isinstance(run_log, dict) else None
    stopped = [(r.get("run"), r.get("status")) for r in (runs or [])
               if isinstance(r, dict) and r.get("status") in STOP_STATUSES]
    if stopped:
        out.append(f"run_log.json records stopped runs {stopped}")
    if isinstance(run_log, dict) and run_log.get("status") in STOP_STATUSES:
        out.append(f"run_log.json status {run_log['status']!r}")
    if isinstance(cfg, dict) and cfg.get("stopped"):
        st = cfg["stopped"]
        out.append(f"config.json records a stop ({st.get('status') if isinstance(st, dict) else st!r})")
    if STOPPED_FILE in names:
        out.append(f"{STOPPED_FILE} present")
    return [p + "; a stopped arm is not a result (PREREG_03 1.3): write a revision first" for p in out]


def arm_structure(c, x, adir, p02_inner_keys, p02_val):
    """Item 1 of the check; returns (stops, record, loaded dumps or None)."""
    stops, rec = [], {"dir": adir}
    missing = [r for r in arm_required(x) if not os.path.exists(os.path.join(adir, *r.split("/")))]
    if missing:
        return [f"arm {x}: missing files {missing}"], dict(rec, missing=missing), None
    cfg = read_json(os.path.join(adir, "config.json"))
    flag = cfg.get("smoke") if isinstance(cfg, dict) else None
    if c.smoke and flag is not True:
        return [f"arm {x}: --smoke needs SMOKE results (config.json smoke {flag!r})"], rec, None
    if not c.smoke and flag is not False:
        return [f"arm {x}: config.json smoke must be false in real mode (got {flag!r})"], rec, None
    arm = config_arm(cfg)
    rec["config_arm"] = arm
    if arm is not None and arm.lower() != x:
        stops.append(f"arm {x}: config.json says ARM {arm!r}")
    run_log = read_json(os.path.join(adir, "run_log.json"))
    status = last_run_status(run_log)
    rec["run_status"] = status
    if status != "completed":
        stops.append(f"arm {x}: last run status {status!r}, not 'completed'")
    earlier = stop_record_problems(run_log, cfg, set(os.listdir(adir)))
    rec["stop_records"] = earlier
    stops.extend(f"arm {x}: {p}" for p in earlier)
    ddir = os.path.join(adir, "dumps")
    for stem, ref, names in (("inner_keys", p02_inner_keys, ("rank", "game_id", "player_id", "color", "label")),
                             ("val_keys", p02_val, ("question_id", "slot", "color", "parsed"))):
        diff = key_array_diff(os.path.join(ddir, f"{stem}.npz"), ref, names)
        rec[f"{stem}_equal_to_prereg02"] = not diff
        if diff:
            stops.append(f"arm {x}: {stem}.npz differs from prereg02's ({diff})")
    keys = load_inner_keys(ddir)
    vk = load_val_keys(ddir)
    check_smoke_size(keys["n"], EXPECTED["inner"]["games"], f"arm {x} inner_keys.npz", c.smoke)
    check_smoke_size(vk["n"], VAL_GAMES_FULL, f"arm {x} val_keys.npz", c.smoke)
    d_in = load_dump(os.path.join(ddir, f"inner_p03{x}_d4.npz"), keys["n"], N_SYMS)
    d_val = load_dump(os.path.join(ddir, f"val_p03{x}_d4.npz"), vk["n"], N_SYMS)
    s_in = load_swap(os.path.join(ddir, f"inner_p03{x}_s0_swap.npz"), keys["n"])
    s_val = load_swap(os.path.join(ddir, f"val_p03{x}_s0_swap.npz"), vk["n"])
    wsha = {d["path"]: d["weights_sha256"] for d in (d_in, d_val, s_in, s_val)}
    rec["weights_sha256"] = d_in["weights_sha256"]
    if len(set(wsha.values())) != 1:
        stops.append(f"arm {x}: the four dumps name different weights {sorted(set(wsha.values()))}")
    rec["config_final_sha256"] = cfg.get("final_sha256")
    rec["config_training_complete"] = cfg.get("training_complete")
    if cfg.get("final_sha256") is not None and cfg["final_sha256"] != d_in["weights_sha256"]:
        stops.append(f"arm {x}: config.json final_sha256 differs from the dumps' weights (final.pth of step T)")
    if cfg.get("training_complete") is False or (not c.smoke and cfg.get("training_complete") is not True):
        stops.append(f"arm {x}: config.json training_complete is {cfg.get('training_complete')!r}, not true")
    markers = {}
    for stem, rows, ws, sy in ((f"inner_p03{x}_d4", keys["n"], d_in["weights_sha256"], N_SYMS),
                               (f"val_p03{x}_d4", vk["n"], d_val["weights_sha256"], N_SYMS),
                               (f"inner_p03{x}_s0_swap", keys["n"], s_in["weights_sha256"], None),
                               (f"val_p03{x}_s0_swap", vk["n"], s_val["weights_sha256"], None),
                               ("inner_keys", keys["n"], None, None), ("val_keys", vk["n"], None, None)):
        p, m = marker_check(ddir, stem, rows, ws, sy)
        stops.extend(f"arm {x}: {q}" for q in p)
        markers[stem] = m
    rec["markers"] = markers
    hist_rows, max_step = read_history(os.path.join(adir, "ckpt", "history.csv"))
    t_cfg, t_where = config_t(cfg)
    want_t = T_STEPS if not c.smoke else t_cfg
    rec["history"] = {"rows": hist_rows, "max_global_step": max_step, "T_required": want_t,
                      "T_in_config": t_cfg, "T_config_key": t_where}
    if not c.smoke and t_cfg is not None and t_cfg != T_STEPS:
        stops.append(f"arm {x}: config.json {t_where} = {t_cfg}, registered T = {T_STEPS}")
    if want_t is None:
        rec["history"]["note"] = "smoke run without a recorded T: step count recorded only"
    elif max_step != want_t:
        stops.append(f"arm {x}: history.csv largest global_step {max_step} != T {want_t} (final.pth must be step T)")
    if not np.isfinite(d_in["logits"]).all():
        stops.append(f"arm {x}: non-finite inner logits (every inner game is parsed)")
    if not np.isfinite(s_in["logits"]).all():
        stops.append(f"arm {x}: non-finite inner swapped logits")
    try:
        check_val_logits(d_val, vk["parsed"])
    except P02Error as e:
        stops.append(f"arm {x}: {e}")
    nan_val = np.isnan(d_val["logits"]).all(axis=(1, 2))
    nan_swap = np.isnan(s_val["logits"]).all(axis=1)
    fin_swap = np.isfinite(s_val["logits"]).all(axis=1)
    rec["val"] = {"shape": list(d_val["logits"].shape), "sha256": d_val["sha256"],
                  "swap_sha256": s_val["sha256"], "nan_rows": int(nan_val.sum())}
    if not np.array_equal(nan_val, p02_val_nan(c)):
        stops.append(f"arm {x}: validation NaN rows differ from val_p02e9_d4's (parse sets must be identical)")
    if not (np.array_equal(nan_swap, nan_val) and fin_swap[~nan_val].all()):
        stops.append(f"arm {x}: swapped validation logits: NaN rows differ or non-finite parsed rows")
    readonly = {}
    for f in sorted(os.listdir(adir)):
        if "readonly_after" in f and f.endswith(".json"):
            runs = (read_json(os.path.join(adir, f)) or {}).get("runs") or []
            readonly[f] = runs[-1].get("same_as_before") if runs else None
            if not runs or runs[-1].get("same_as_before") is not True:
                stops.append(f"arm {x}: {f} does not say the read-only folders were unchanged (last run)")
    rec["readonly_after"] = readonly
    rec["inputs_sha256"] = {os.path.basename(p): sha256_file(p) for p in
                            [os.path.join(ddir, f"{s}.npz") for s in arm_stems(x)]}
    return stops, rec, {"keys": keys, "vk": vk, "inner": d_in, "val": d_val, "inner_swap": s_in, "val_swap": s_val}


_P02_VAL_NAN = {}


def p02_val_nan(c):
    key = _real(c.p02_dumps)
    if key not in _P02_VAL_NAN:
        vk = load_val_keys(c.p02_dumps)
        d = load_dump(os.path.join(c.p02_dumps, f"val_{C5_DUMP}.npz"), vk["n"], N_SYMS)
        _P02_VAL_NAN[key] = np.isnan(d["logits"]).all(axis=(1, 2))
    return _P02_VAL_NAN[key]


def sym_accuracies(logits, label):
    return [float(np.mean(np.argmax(logits[:, s, :], axis=1) == label)) * 100 for s in range(logits.shape[1])]


# ================================================================ unpack
def member_problem(name, prefix):
    if name.startswith("/") or "\\" in name or ":" in name:
        return f"{name}: absolute path or backslash"
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts[:-1]) or parts[-1] in (".", ".."):
        return f"{name}: '.' or '..' segment"
    if not name.startswith(prefix):
        return f"{name}: outside {prefix}"
    base = parts[-1].lower()
    if "answer" in base:
        return f"{name}: label-file name"
    if not base.endswith(ZIP_ALLOWED_EXT):
        return f"{name}: file type not allowed (weights, state, temporary and archive files are never unpacked)"
    return None


def cmd_unpack(a):
    x = a.arm
    zpath = os.path.abspath(a.zip)
    refuse_label_file(zpath)
    if "failed" in os.path.basename(zpath).lower():
        raise P03Error(f"{zpath}: a *failed* result zip is never unpacked (the arm is not a result)")
    if not zipfile.is_zipfile(zpath):
        raise P03Error(f"{zpath} is not a zip file")
    imported_from_tools()
    frozen = frozen_tools_check()
    real_dest = os.path.join(P03_ROOT, f"rank_{x}")
    if a.smoke:
        if not a.dest:
            raise P03Error("--smoke unpack needs --dest outside the repository")
        dest = os.path.abspath(a.dest)
        if is_inside(dest, REPO):
            raise P03Error(f"--smoke --dest must be outside the repository (got {dest})")
    else:
        if a.dest and not same_path(a.dest, real_dest):
            raise P03Error(f"real mode unpacks into {real_dest} only")
        dest = real_dest
    for ro in READONLY_ROOTS:
        if is_inside(dest, os.path.join(RESULTS, ro)):
            raise P03Error(f"refusing to unpack into results/{ro}")
    record = os.path.join(os.path.dirname(dest), f"unpack_rank_{x}.json")
    if (os.path.isdir(dest) and os.listdir(dest)) or os.path.exists(record) or os.path.isfile(dest):
        raise P03Error(f"{dest} (or {record}) already exists; nothing extracted (a rerun needs a revision first)")
    prefix = f"rank_{x}/"
    with zipfile.ZipFile(zpath) as z:
        infos = [i for i in z.infolist() if not i.filename.endswith("/")]
        names = [i.filename for i in infos]
        problems = [p for p in (member_problem(n, prefix) for n in names) if p]
        if len(set(names)) != len(names):
            problems.append("repeated members")
        problems += [f"{i.filename}: {i.file_size} bytes (>= 100 MB)" for i in infos if i.file_size >= ZIP_MAX_BYTES]
        rels = [n[len(prefix):] for n in names]
        missing = [r for r in arm_required(x) if r not in rels]
        if problems or missing:
            raise P03Error(f"{zpath}: refused; problems {problems[:6]}, missing {missing}")
        cfg = json.loads(z.read(prefix + "config.json").decode("utf-8"))
        flag = cfg.get("smoke") if isinstance(cfg, dict) else None
        if a.smoke and flag is not True:
            raise P03Error(f"--smoke unpack takes SMOKE results only (config.json smoke {flag!r})")
        if not a.smoke and flag is not False:
            raise P03Error(f"real mode takes real results only (config.json smoke {flag!r})")
        run_log = json.loads(z.read(prefix + "run_log.json").decode("utf-8"))
        status = last_run_status(run_log)
        if status != "completed":
            raise P03Error(f"{zpath}: last run status {status!r}, not 'completed'; the arm is not a result")
        earlier = stop_record_problems(run_log, cfg, set(rels))
        if earlier:
            raise P03Error(f"{zpath}: refused; {'; '.join(earlier)}")
        member_sha = {}
        for info in infos:
            h = hashlib.sha256()
            with z.open(info) as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            member_sha[info.filename[len(prefix):]] = h.hexdigest()
        bad = []
        for s in arm_stems(x):
            mk = json.loads(z.read(prefix + f"dumps/_DONE_{s}.json").decode("utf-8"))
            if mk.get("sha256") != member_sha[f"dumps/{s}.npz"]:
                bad.append(s)
        if bad:
            raise P03Error(f"{zpath}: completion marker SHA-256 differs from the dump for {bad}")
        os.makedirs(dest, exist_ok=True)
        for info in infos:
            target = os.path.join(dest, *info.filename[len(prefix):].split("/"))
            if not is_inside(target, dest):
                raise P03Error(f"{info.filename}: escapes {dest}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target + ".tmp", "wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    dst.write(chunk)
            os.replace(target + ".tmp", target)
    for rel, h in member_sha.items():
        if sha256_file(os.path.join(dest, *rel.split("/"))) != h:
            raise P03Error(f"{rel}: extracted file differs from the zip member")
    rec = {"format": UNPACK_FORMAT, "created": now_iso(), "arm": x, "smoke": bool(a.smoke),
           "zip": zpath, "zip_name": os.path.basename(zpath), "zip_sha256": sha256_file(zpath),
           "zip_bytes": os.path.getsize(zpath), "dest": dest, "members": member_sha,
           "run_status": status, "config_smoke": flag, "tool": tool_state(), "frozen_tools": frozen}
    write_json(record, rec)
    print(f"unpack arm {x}: {len(member_sha)} files from {zpath} -> {dest}; record {record}")
    if not a.smoke:
        print(f"next: commit results/prereg03/rank_{x}/ and results/prereg03/unpack_rank_{x}.json, then 'check'")
    return 0


def arms_committed(c, arms):
    if c.smoke or not arms:
        return None
    paths = []
    for x in arms:
        paths += files_under(c.arm_dir[x]) + [os.path.join(c.p03_dir, f"unpack_rank_{x}.json")]
    return require_tracked_clean(paths, "unpacked arm results (commit them before check / tune / apply)")


# ================================================================ check
def cmd_check(a):
    t0 = time.time()
    c = resolve(a, "check")
    frozen = start(c)
    p02 = p02_inputs(c, val=True)
    present = [x for x in ARMS if os.path.isdir(c.arm_dir[x]) and os.listdir(c.arm_dir[x])]
    commit = arms_committed(c, present)
    inner = prepare_inner(c)
    keys = inner["keys"]
    ms = mono_set(inner)   # single-colour inner pseudo-questions (record only, PREREG_03 rev 3)
    if not ms["questions"] and not c.smoke:
        raise P03Error("no single-colour inner pseudo-question (the real inner data always has them)")
    mono_boot = PlayerBootstrap(ms["player"], ms["y"], ms["n_players"]) if ms["questions"] else None
    c5 = load_c5_inner(c, keys["n"])
    label = keys["label"]
    c5_acc = sym_accuracies(c5["logits"], label)
    sources = {"C5": {"status": "usable", "text": SOURCE_TEXT["C5"], "inner_accuracy_percent": c5_acc[0],
                      "per_symmetry_accuracy_percent": c5_acc, "weights_sha256": c5["weights_sha256"],
                      "note": "committed prereg02 source (R20); usable by registration"}}
    all_stops = []
    val_groups = None
    for x in ARMS:
        s = ARM_SOURCE[x]
        if x not in present:
            sources[s] = {"status": "not_run", "text": SOURCE_TEXT[s],
                          "note": "no unpacked result (arm not run or dropped): its combinations are dropped"}
            continue
        stops, rec, ld = arm_structure(c, x, c.arm_dir[x], os.path.join(c.p02_dumps, "inner_keys.npz"),
                                       os.path.join(c.p02_dumps, "val_keys.npz"))
        rec["text"] = SOURCE_TEXT[s]
        if ld is None or stops:
            rec.update(status="stop", stops=stops)
            sources[s] = rec
            all_stops.extend(stops)
            continue
        z = ld["inner"]["logits"]
        accs = sym_accuracies(z, label)
        acc_c = accs[0]
        acc_s = float(np.mean(np.argmax(ld["inner_swap"]["logits"], axis=1) == label)) * 100
        floor_ok = acc_c >= c.min_acc
        # kappa on the inner eval rounds and on the validation questions (>= 2 parsed games)
        lp_c = log_softmax(z[:, 0, :].astype(np.float64))
        lp_s = log_softmax(ld["inner_swap"]["logits"].astype(np.float64))
        kc, ks = kappa_questions(lp_c, inner["eval_groups"]), kappa_questions(lp_s, inner["eval_groups"])
        dk = kc - ks
        if np.isnan(dk).any():
            raise P03Error("inner eval questions with < 2 games (every eval question has >= 5)")
        draws = boot_plain_mean(inner["boot"], dk)
        dk_in = float(np.mean(dk))
        ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        rb_draws = inner["boot"].stats(dk)
        vk = ld["vk"]
        if val_groups is None:
            val_groups = [[i for i in rows if vk["parsed"][i]] for _, rows in vk["groups"]]
        vz = np.where(vk["parsed"][:, None], ld["val"]["logits"][:, 0, :], 0.0).astype(np.float64)
        vs = np.where(vk["parsed"][:, None], ld["val_swap"]["logits"], 0.0).astype(np.float64)
        kvc, kvs = kappa_questions(log_softmax(vz), val_groups), kappa_questions(log_softmax(vs), val_groups)
        used = ~np.isnan(kvc)
        dk_val = float(np.mean(kvc[used] - kvs[used])) if used.any() else None
        stops, colour = colour_decision(acc_c, acc_s, ci[0], dk_val)
        if c.record_colour and stops:
            # SMOKE only: an untrained SMOKE model's colour signal is noise (400 inner games, 1 point = 4 games),
            # so the stops are computed and recorded but do not stop the smoke chain
            colour["stops_recorded_only_smoke"] = stops
            stops = []
        colour.update({
            "accuracy_correct_percent": acc_c, "accuracy_swapped_percent": acc_s,
            "kappa_inner": {"correct": float(np.mean(kc)), "swapped": float(np.mean(ks)), "dkappa": dk_in,
                            "ci95": ci, "questions": int(len(dk)),
                            "bootstrap": f"PlayerBootstrap draws (players within each level, {N_BOOT}, "
                                         f"default_rng({BOOT_SEED})) of the plain question mean",
                            "rank_balanced_dkappa_ci95_recorded_only": [float(np.percentile(rb_draws, 2.5)),
                                                                        float(np.percentile(rb_draws, 97.5))]},
            "kappa_val": {"correct": float(np.mean(kvc[used])) if used.any() else None,
                          "swapped": float(np.mean(kvs[used])) if used.any() else None,
                          "dkappa": dk_val, "questions": int(used.sum())},
            "definition": "kappa = mean over questions of (mean over games g of sum_k softmax(mean_{h != g} "
                          "l_h)_k l_g,k), l = original-orientation log_softmax; questions with >= 2 games"})
        colour["kappa_single_colour"] = kappa_single_colour(lp_c, lp_s, ms, mono_boot, colour["dacc_points"], vk,
                                                            log_softmax(vz), log_softmax(vs))
        hist = rec["history"]["rows"]
        gap = None
        if hist:
            last = hist[-1]
            inner_col = next((k for k in ("inner_acc", "val_acc") if last.get(k) not in (None, "")), None)
            try:
                if inner_col and last.get("train_acc") not in (None, ""):
                    gap = {"train_acc": float(last["train_acc"]), inner_col: float(last[inner_col]),
                           "gap_points": float(last["train_acc"]) - float(last[inner_col])}
            except ValueError:
                gap = None
        rec.update({
            "status": "stop" if stops else ("usable" if floor_ok else "unusable"),
            "stops": stops,
            "floor": {"inner_accuracy_percent": acc_c, "threshold_percent": c.min_acc, "passed": bool(floor_ok),
                      "registered_threshold_percent": P02E9_MIN_ACC},
            "colour": colour,
            "recorded": {"per_symmetry_accuracy_percent": accs,
                         "per_symmetry_spread_points": max(accs) - min(accs),
                         "C5_per_symmetry_accuracy_percent": c5_acc,
                         "history_last_row": hist[-1] if hist else None,
                         "train_minus_inner_last_pass": gap,
                         "history_passes": len(hist)}})
        sources[s] = rec
        all_stops.extend(stops)
    usable = [s for s in ("C5", "C6A", "C6B") if sources[s]["status"] == "usable"]
    combos = [{"id": cid, "sources": list(srcs), "status": st, "reason": why}
              for cid, srcs, st, why in available_combos(usable)]
    vk02 = load_val_keys(c.p02_dumps)
    vsub = val_colour_subquestions(vk02)
    single = {"what": "PREREG_03 rev 3, recorded only: single-colour kappa reference (per arm: sources.<C6x>.colour."
                      "kappa_single_colour); the check's stops and usable sources do not depend on it",
              "inner_questions": {"definition": ms["definition"], "questions_sha256": ms["sha256"],
                                  "counts": ms["counts"]},
              "val_sub_questions": {"sub_questions": len(vsub),
                                    "by_colour": {col: sum(1 for _, cc, _ in vsub if cc == col) for col in MONO_COLOURS},
                                    "questions_with_a_sub_question": len({q for q, _, _ in vsub}),
                                    "val_questions": len(vk02["groups"])}}
    rep = {"format": CHECK_FORMAT,
           "what": "PREREG_03 2.2 items 1-4 (plan.md 4.4): can the new sources be used (before any fitting)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), "smoke": c.smoke, "preview": c.preview,
           "status": "stop" if all_stops else "ok", "stops": all_stops, "usable_sources": usable,
           "sources": sources, "combinations": combos, "single_colour": single,
           "thresholds": {"floor_percent": c.min_acc, "wiring_stop_points": WIRING_STOP_POINTS,
                          "kappa_min_dacc_points": KAPPA_MIN_DACC_POINTS, "T": T_STEPS,
                          "colour_stops": "recorded only (--smoke-record-colour)" if c.record_colour else "applied"},
           "p02_inputs": p02, "inner_inputs_sha256": inner["inputs"],
           "arms_commit_check": commit, "tool": tool_state(), "frozen_tools": frozen,
           "environment": environment()}
    os.makedirs(c.out_dir, exist_ok=True)
    write_json(os.path.join(c.out_dir, "check_report.json"), rep)
    for s in ("C6A", "C6B"):
        r = sources[s]
        extra = ""
        if "floor" in r:
            extra = (f": inner accuracy {r['floor']['inner_accuracy_percent']:.3f}%, dacc "
                     f"{r['colour']['dacc_points']:+.3f} points, dkappa inner {r['colour']['kappa_inner']['dkappa']:+.4g}"
                     f" CI [{r['colour']['kappa_inner']['ci95'][0]:+.3g}, {r['colour']['kappa_inner']['ci95'][1]:+.3g}]"
                     f", val {r['colour']['kappa_val']['dkappa']}")
            mono = r["colour"]["kappa_single_colour"]
            if mono["inner"]:
                extra += (f"; single-colour (record only) dkappa inner {mono['inner']['dkappa']:+.4g} CI "
                          f"[{mono['inner']['ci95'][0]:+.3g}, {mono['inner']['ci95'][1]:+.3g}], val sub-questions "
                          f"{(mono['val_sub_questions'] or {}).get('dkappa')}")
        print(f"{s}: {r['status']}{extra}")
    print(f"check: usable sources {usable}; status {rep['status']}{' (preview)' if c.preview else ''} -> "
          f"{os.path.join(c.out_dir, 'check_report.json')}")
    for s in all_stops:
        print("STOP: " + s, file=sys.stderr)
    return 1 if all_stops else 0


# ================================================================ selfcheck
def compare_oof(mine, ref, path=""):
    """[(path, detail)] where two OOF records differ: floats > 1e-12, everything else exactly."""
    out = []
    if isinstance(ref, dict):
        for k, v in ref.items():
            if k.startswith("diff_vs_"):
                continue   # prereg02's R00 comparison, not part of P00
            if k not in mine:
                out.append((path + "/" + k, "missing"))
            else:
                out.extend(compare_oof(mine[k], v, path + "/" + k))
    elif isinstance(ref, float) or isinstance(mine, float):
        if not isinstance(mine, (int, float)) or not isinstance(ref, (int, float)) or \
                abs(float(mine) - float(ref)) > SELFCHECK_TOL:
            out.append((path, f"{mine!r} vs {ref!r}"))
    elif isinstance(ref, list):
        if not isinstance(mine, list) or len(mine) != len(ref):
            out.append((path, "length"))
        else:
            for i, (u, v) in enumerate(zip(mine, ref)):
                out.extend(compare_oof(u, v, f"{path}[{i}]"))
    elif mine != ref:
        out.append((path, f"{mine!r} vs {ref!r}"))
    return out


def fold_diffs(mine, ref):
    out = []
    if len(mine) != len(ref):
        return [("folds", "count")]
    for f, (u, v) in enumerate(zip(mine, ref)):
        if (v.get("params") is None) != (u.get("params") is None):
            out.append((f"fold {f} params", "presence"))
            continue
        if v.get("params") is not None:
            d = params_diff(u["params"], v["params"])
            if d > SELFCHECK_TOL:
                out.append((f"fold {f} params", d))
        for k in ("loss",):
            if k in v and abs(u.get(k, float("nan")) - v[k]) > SELFCHECK_TOL:
                out.append((f"fold {f} {k}", f"{u.get(k)} vs {v[k]}"))
        for k in ("fit_questions", "eval_questions", "fold"):
            if k in v and u.get(k) != v[k]:
                out.append((f"fold {f} {k}", f"{u.get(k)} vs {v[k]}"))
    return out


def cmd_selfcheck(a):
    t0 = time.time()
    c = resolve(a, "selfcheck")
    frozen = start(c)
    order = None
    if not c.smoke and c.in_repo:
        order = require_tracked_clean([os.path.join(c.out_dir, "check_report.json")],
                                      "check_report.json (run and commit 'check' before 'selfcheck')")
        cr = read_json(os.path.join(c.out_dir, "check_report.json"))
        if cr.get("status") != "ok" or cr.get("preview") or cr.get("smoke"):
            raise P03Error("check_report.json is not a passed real check (status ok, no preview)")
    elif c.smoke:
        crp = os.path.join(c.out_dir, "check_report.json")
        if os.path.exists(crp) and read_json(crp).get("status") != "ok":
            raise P03Error("check_report.json in the smoke out-dir did not pass")
    p02 = p02_inputs(c, val=True)
    inner = prepare_inner(c)
    keys = inner["keys"]
    c5 = load_c5_inner(c, keys["n"])
    B = basis()
    logits = {C5_DUMP: c5["logits"]}
    lp = combo_logprobs(("C5",), logits)
    lp_ref = rf.source_logprobs("C5", logits)
    lp_equal = bool(np.array_equal(lp, lp_ref))
    add_combo(inner, "P00", lp)
    m, sc, folds, _ = run_oof(inner, "P00", True, c.smoke, B)
    m.update(inner["boot"].summary(sc))
    theta, refit = refit_all(inner, "P00", True, c.smoke, B)
    par = params_dict(theta, True, B)
    checks, failures = {}, []

    def need(name, ok, detail, required=True):
        checks[name] = {"ok": ok, "detail": detail, "required": required}
        if required and ok is not True:
            failures.append(name)

    need("per_game_logprobs_equal_p02_source_logprobs_C5", lp_equal,
         "combo_logprobs(('C5',)) vs p02_rank_fuse.source_logprobs('C5'), element by element")
    # 1. inner OOF vs tune_report.json R20
    tpath = os.path.join(c.p02_local, "tune_report.json")
    tr = read_json(tpath) if os.path.exists(tpath) else None
    r20 = next((x for x in (tr or {}).get("combinations", []) if x.get("id") == "R20"), None)
    avail = r20 is not None and r20.get("status") == "ok"
    if avail:
        d_oof = compare_oof(m, r20["oof"])
        d_fold = fold_diffs(folds, r20["folds"])
        need("inner_oof_equals_R20", not d_oof and not d_fold,
             {"P00_rank_balanced": m["rank_balanced"], "R20_rank_balanced": r20["oof"]["rank_balanced"],
              "abs_diff": abs(m["rank_balanced"] - r20["oof"]["rank_balanced"]),
              "differences": [list(map(str, d)) for d in (d_oof + d_fold)][:20],
              "compared": "every entry of the R20 OOF record (rank_balanced, question_mean, n_questions, "
                          "levels_present, per_level, per_n_bin, pred_distribution, se) and every fold's "
                          "params, loss and counts; floats <= 1e-12, the rest exactly"})
        if not c.smoke:
            need("registered_R20_value", r20["oof"]["rank_balanced"] == R20_INNER_OOF,
                 {"tune_report": r20["oof"]["rank_balanced"], "registered": R20_INNER_OOF})
    else:
        need("inner_oof_equals_R20", None, "tune_report.json with an R20 entry not available", required=not c.smoke)
    # 2. refit vs frozen_config.json
    fpath = os.path.join(c.p02_local, "frozen_config.json")
    fc = read_json(fpath) if os.path.exists(fpath) else None
    fc_r20 = fc is not None and (fc.get("combo") or {}).get("id") == "R20"
    if fc_r20:
        pd = params_diff(par, fc["params"])
        need("refit_params_equal_frozen", pd <= SELFCHECK_TOL,
             {"max_abs_diff": pd, "tolerance": SELFCHECK_TOL, "refit_loss": refit.get("loss"),
              "frozen_refit_loss": ((tr or {}).get("refit") or {}).get("loss")})
    else:
        need("refit_params_equal_frozen", None, "frozen_config.json of R20 not available", required=not c.smoke)
    # 3. validation probabilities vs rank_val_probs.csv and pred_B.csv
    vpath = os.path.join(c.p02_local, "rank_val_probs.csv")
    if fc_r20 and os.path.exists(vpath):
        vk, vlog, vsha, _ = load_val_side(c, ("C5",), c.arm_dir)
        op_o, op_has, _op = opening_val_aligned(c, vk, fc.get("opening_build_id"))
        P, n, ok = val_probs(("C5",), theta, True, B, vk, vlog, op_o, op_has)
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
        pb = os.path.join(c.p02_local, "pred_B.csv")
        pb_diff = None
        if os.path.exists(pb):
            with open(pb, newline="", encoding="utf-8") as f:
                saved = {r["question_id"]: r["pred_rank"] for r in csv.DictReader(f)}
            pb_diff = [q for k, q in enumerate(qids) if ok[k] and saved.get(q) != decide.rule_b([float(x) for x in P[k]])]
        need("val_probs_equal_rank_val_probs", not row_bad and max_diff <= SELFCHECK_TOL and not b_diff
             and not pb_diff,
             {"questions": len(qids), "max_abs_diff": max_diff, "tolerance": SELFCHECK_TOL,
              "row_mismatches": row_bad[:10], "rule_b_differences": b_diff[:10],
              "pred_B_differences": pb_diff if pb_diff is None else pb_diff[:10],
              "inputs_sha256": dict(vsha, **{"rank_val_probs.csv": sha256_file(vpath),
                                            "pred_B.csv": sha256_file(pb) if os.path.exists(pb) else None})})
    else:
        need("val_probs_equal_rank_val_probs", None, "R20's rank_val_probs.csv / frozen_config.json not available",
             required=not c.smoke)
    passed = not failures
    rep = {"format": SELFCHECK_FORMAT,
           "what": "PREREG_03 2.2 item 5 (plan.md 4.5 step 3): P00 = C5 through prereg03's path reproduces "
                   "prereg02 R20 (inner OOF, refit, validation probabilities)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), "smoke": c.smoke, "preview": c.preview,
           "passed": passed, "failures": failures, "checks": checks,
           "P00": {"oof": m, "folds": folds, "refit": refit, "params": par},
           "order_check": order, "p02_inputs": p02, "inner_inputs_sha256": dict(inner["inputs"], **{
               f"inner_{C5_DUMP}.npz": c5["sha256"]}),
           "reference_files_sha256": {k: (sha256_file(p) if os.path.exists(p) else None) for k, p in (
               ("tune_report.json", tpath), ("frozen_config.json", fpath), ("rank_val_probs.csv", vpath))},
           "tool": tool_state(), "frozen_tools": frozen, "environment": environment()}
    os.makedirs(c.out_dir, exist_ok=True)
    out = os.path.join(c.out_dir, "selfcheck.json")
    write_json(out, rep)
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v['ok'] else ('n/a' if v['ok'] is None else 'FAILED')}")
    print(f"selfcheck: P00 inner OOF {m['rank_balanced']!r} (R20 {R20_INNER_OOF!r}) -> "
          f"{'PASSED' if passed else 'FAILED'}{' (preview; tune accepts only the committed real one)' if c.preview else ''}"
          f" ({time.time() - t0:.0f}s) -> {out}")
    return 0 if passed else 1


# ================================================================ tune
def load_r10(c):
    if not c.fallback_r10 or not os.path.exists(c.fallback_r10):
        if not c.smoke:
            raise P03Error(f"missing {c.fallback_r10}")
        return None
    obj = read_json(c.fallback_r10)
    want = {"id": "R10", "source": "C5", "pool": "log", "calibrated": True, "opening": False}
    if obj.get("format") != R10_FORMAT or {k: (obj.get("combo") or {}).get(k) for k in want} != want:
        raise P03Error(f"{c.fallback_r10}: not the registered R10 fallback")
    v = obj.get("inner_oof_rank_balanced")
    if not isinstance(v, float):
        raise P03Error(f"{c.fallback_r10}: no inner_oof_rank_balanced")
    rec = {"path": c.fallback_r10, "sha256": sha256_file(c.fallback_r10), "inner_oof_rank_balanced": v,
           "params": obj.get("params")}
    if not c.smoke:
        if round(v, 4) != R10_INNER_REGISTERED:
            raise P03Error(f"R10 inner OOF {v} is not the registered {R10_INNER_REGISTERED}")
        rec["commit_check"] = require_tracked_clean([c.fallback_r10], "fallback_R10.json")
    return rec


def single_colour_tune_records(ms, records, scores, boot, cid, cr):
    """tune_report.json / frozen_config.json records of the single-colour inner pseudo-questions and the
    single-colour kappa reference of the selected sources (PREREG_03 rev 3; recorded only)."""
    base = {"what": "PREREG_03 rev 3, recorded only: 5-fold OOF on the single-colour inner pseudo-questions (fold f's "
                    "players scored with the parameters fitted on the other folds' registered mixed-colour fit "
                    "questions); never used for the selection, the gates, the refit, the FAIL_OPENING choice or H3R",
            "questions_sha256": ms["sha256"], "counts": ms["counts"]}
    kref = {}
    for s in combo_sources(cid):
        if s in SOURCE_ARM:
            col = ((cr.get("sources") or {}).get(s) or {}).get("colour") or {}
            kref[s] = {"dacc_points": col.get("dacc_points"),
                       "kappa_mixed_inner_dkappa": (col.get("kappa_inner") or {}).get("dkappa"),
                       "kappa_mixed_val_dkappa": (col.get("kappa_val") or {}).get("dkappa"),
                       "kappa_single_colour": col.get("kappa_single_colour")}
    if not ms["questions"]:
        status = "no single-colour inner pseudo-question (smoke data only; the real inner data always has them)"
        return dict(base, status=status, definition=ms["definition"]), dict(base, status=status, kappa_reference=kref)

    def short(r):
        return {"rank_balanced": r["rank_balanced"], "question_mean": r["question_mean"], "se": r.get("se"),
                "n_questions": r["n_questions"], "per_level": r["per_level"],
                "by_colour": {c: {k: v[k] for k in ("rank_balanced", "question_mean", "n_questions")}
                              for c, v in r["by_colour"].items()},
                "by_k": {k: {kk: v[kk] for kk in ("rank_balanced", "question_mean", "n_questions")}
                         for k, v in r["by_k"].items()}}
    diff = mono_diff_record(ms, boot, scores[cid], scores["P00"]) if cid != "P00" else None
    rep = dict(base, status="ok", definition=ms["definition"], combinations=dict(records), selected=cid,
               selected_minus_P00=diff,
               note="every fitted combination is recorded; the fold records' parameters were checked to reproduce "
                    "the eval-round OOF probabilities exactly before scoring")
    frz = dict(base, status="ok", selected=dict({"id": cid}, **short(records[cid])), P00=short(records["P00"]),
               selected_minus_P00=diff, kappa_reference=kref)
    return rep, frz


def single_colour_score_block(cid, kref):
    """PREREG_03 rev 3 11(c), second case, already known when tune runs: a C6 source of the selected combination whose
    single-colour condition holds (dacc >= 0.5 points and single-colour inner dkappa CI lower bound > 0) while its
    validation single-colour sub-question dkappa is not > 0 means 03 cannot be used on test day, so H3R is not run and
    the scoring chance is kept. A C6 source without a single-colour reference blocks too. -> [reasons] ([] for P00)."""
    if cid == "P00":
        return []
    out = []
    for s in combo_sources(cid):
        if s not in SOURCE_ARM:
            continue
        ref = (((kref or {}).get(s) or {}).get("kappa_single_colour") or {}).get("test_day_reference")
        if not isinstance(ref, dict):
            out.append(f"{s}: no single-colour kappa reference (PREREG_03 rev 3 11(c)); H3R is not run, the scoring "
                       f"chance is kept")
        elif ref.get("single_colour_condition") is True and ref.get("val_sub_question_dkappa_positive") is not True:
            out.append(f"{s}: single-colour condition holds (dacc {ref.get('dacc_points')} points, single-colour inner "
                       f"dkappa CI lower bound {ref.get('inner_dkappa_ci95_lower')}) but the validation single-colour "
                       f"sub-question dkappa is {ref.get('val_sub_question_dkappa')} (not > 0): 03 cannot be used on "
                       f"test day (PREREG_03 rev 3 11(c)); H3R is not run, the scoring chance is kept")
    return out


def require_order(c, names):
    """Earlier steps: the files exist in the out-dir; real mode: tracked and unchanged."""
    paths = [os.path.join(c.out_dir, n) for n in names]
    missing = [n for n, p in zip(names, paths) if not os.path.exists(p)]
    if missing:
        raise P03Error(f"run the earlier steps first: {missing} missing in {c.out_dir}")
    if c.smoke:
        return None
    return require_tracked_clean(paths, f"{names} (commit each step before the next)")


def cmd_tune(a):
    t0 = time.time()
    c = resolve(a, "tune")
    frozen_path = os.path.join(c.out_dir, "frozen_config.json")
    if not c.smoke:
        if not c.in_repo:
            raise P03Error(f"tune writes the frozen config into {P03_OUT} (committed steps only; no preview)")
        rc, _ = git(["ls-files", "--error-unmatch", "--", repo_rel(frozen_path)])
        if rc == 0:
            raise P03Error("frozen_config.json is already tracked by git (frozen); tune will not overwrite it")
    frozen = start(c)
    tools = tool_state()
    if not c.smoke:
        require_tools_committed(tools)
    order = require_order(c, ["check_report.json", "selfcheck.json"])
    cr = read_json(os.path.join(c.out_dir, "check_report.json"))
    sc_rep = read_json(os.path.join(c.out_dir, "selfcheck.json"))
    if cr.get("format") != CHECK_FORMAT or cr.get("status") != "ok":
        raise P03Error("check_report.json did not pass (status must be ok; a stop needs a fix and a revision)")
    if sc_rep.get("format") != SELFCHECK_FORMAT or sc_rep.get("passed") is not True:
        raise P03Error("selfcheck.json did not pass; tune does not run")
    for rep, what in ((cr, "check_report.json"), (sc_rep, "selfcheck.json")):
        if bool(rep.get("smoke")) != c.smoke or rep.get("preview"):
            raise P03Error(f"{what} is a {'preview' if rep.get('preview') else 'smoke/real'} record that does not "
                           f"match this run")
    usable = list(cr["usable_sources"])
    arms_used = [SOURCE_ARM[s] for s in usable if s in SOURCE_ARM]
    commit = arms_committed(c, arms_used)
    p02 = p02_inputs(c, val=False)   # inner files only: tune reads no val_* file
    inner = prepare_inner(c)
    if inner["inputs"] != cr["inner_inputs_sha256"] or inner["inputs"] != {
            k: v for k, v in sc_rep["inner_inputs_sha256"].items() if k in inner["inputs"]}:
        raise P03Error("inner inputs changed since check / selfcheck")
    keys = inner["keys"]
    loaded = {C5_DUMP: load_c5_inner(c, keys["n"])}
    if loaded[C5_DUMP]["sha256"] != sc_rep["inner_inputs_sha256"].get(f"inner_{C5_DUMP}.npz"):
        raise P03Error("inner_p02e9_d4.npz changed since selfcheck")
    for s in usable:
        if s == "C5":
            continue
        x = SOURCE_ARM[s]
        ddir = os.path.join(c.arm_dir[x], "dumps")
        diff = key_array_diff(os.path.join(ddir, "inner_keys.npz"), os.path.join(c.p02_dumps, "inner_keys.npz"),
                              ("rank", "game_id", "player_id", "color", "label"))
        if diff:
            raise P03Error(f"arm {x}: inner_keys differ from prereg02's {diff}")
        name = SOURCES[s][0][0]
        d = load_dump(os.path.join(ddir, f"inner_{name}.npz"), keys["n"], N_SYMS)
        if d["sha256"] != (cr["sources"][s].get("inputs_sha256") or {}).get(f"inner_{name}.npz"):
            raise P03Error(f"inner_{name}.npz changed since check")
        if not np.isfinite(d["logits"]).all():
            raise P03Error(f"{d['path']}: non-finite logits")
        loaded[name] = d
    logits = {n: d["logits"] for n, d in loaded.items()}
    X_eval = inner["X_eval"]
    boot = inner["boot"]
    B = basis()
    print(f"inputs ready: fit {len(inner['X_fit'])} questions, eval {len(X_eval)} questions, usable sources "
          f"{usable} ({time.time() - t0:.0f}s)", flush=True)
    # single-colour inner pseudo-questions (PREREG_03 rev 3): the same set check used; scored with each fold's OOF
    # parameters, recorded only (the selection, the gates and the frozen parameters never see them)
    ms = mono_set(inner)
    cr_single = (cr.get("single_colour") or {}).get("inner_questions") or {}
    if cr_single.get("questions_sha256") != ms["sha256"]:
        raise P03Error("check_report.json has no single-colour record or it was built from other single-colour "
                       "questions (PREREG_03 rev 3): rerun check with this tool")
    if ms["questions"]:
        X_mono, S_mono = mono_qset(inner, ms)
        mono_boot = PlayerBootstrap(ms["player"], ms["y"], ms["n_players"])
    else:
        X_mono = S_mono = mono_boot = None
    mono_scores, mono_records = {}, {}
    results, scores = [], {}
    for cid, srcs, status, why in available_combos(usable):
        rec = {"id": cid, "sources": list(srcs), "pool": POOL, "calibrated": True, "opening": True}
        if status != "ok":
            rec.update(status="dropped", reason=why)
            results.append(rec)
            continue
        lp = combo_logprobs(srcs, logits)
        add_combo(inner, cid, lp)
        if X_mono is not None:
            X_mono.m[(cid, POOL)] = pooled(S_mono, lp, POOL)
        del lp
        m, sc, folds, oof_P = run_oof(inner, cid, True, c.smoke, B)
        if X_mono is not None:
            # the fold records' parameters reproduce the eval-round OOF probabilities exactly, then score the
            # single-colour questions with them
            if not np.array_equal(fold_predict(X_eval, cid, folds, True, B), oof_P, equal_nan=True):
                raise P03Error(f"{cid}: the fold records' parameters do not reproduce the OOF probabilities")
            P_mono = fold_predict(X_mono, cid, folds, True, B)
            if np.isnan(P_mono).any():
                raise P03Error(f"{cid}: a single-colour question without an OOF fold")
            mono_records[cid], mono_scores[cid] = mono_score_record(rule_b_np(P_mono), X_mono, ms, mono_boot)
        scores[cid] = sc
        m.update(boot.summary(sc, None if cid == "P00" else scores["P00"]))
        for k in ("diff_vs_R00_se", "diff_vs_R00_ci95"):
            if k in m:
                m[k.replace("R00", "P00")] = m.pop(k)
        if cid != "P00":
            m["question_mean_diff_vs_P00"] = float(np.mean(sc - scores["P00"]))
        rec.update(status="ok", oof=m, folds=folds)
        results.append(rec)
        print(f"{cid} {'+'.join(srcs):12s}: OOF rank-balanced {m['rank_balanced']:.4f}, question mean "
              f"{m['question_mean']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    by_id = {r["id"]: r for r in results}
    p00 = by_id["P00"]
    if abs(p00["oof"]["rank_balanced"] - sc_rep["P00"]["oof"]["rank_balanced"]) > SELFCHECK_TOL:
        raise P03Error("P00 differs from the selfcheck's P00")
    chosen, selection = select(results)
    cid = chosen["id"]
    gates = gate_record(cid, chosen["oof"]["rank_balanced"], p00["oof"]["rank_balanced"], scores[cid],
                        scores["P00"], boot)
    delta = chosen["oof"]["rank_balanced"] - p00["oof"]["rank_balanced"]
    d_q = scores[cid] - scores["P00"]
    se = se_val_record(delta, d_q, X_eval.y) if cid != "P00" else None
    per_level_diff = {lv: chosen["oof"]["per_level"][lv] - p00["oof"]["per_level"][lv]
                      for lv in chosen["oof"]["per_level"]}
    theta, refit = refit_all(inner, cid, True, c.smoke, B)
    final = params_dict(theta, True, B)
    # no-opening version of the selected combination (w = 0): same OOF and refit procedure
    m0, _sc0, folds0, _ = run_oof(inner, cid, False, c.smoke, B)
    theta0, refit0 = refit_all(inner, cid, False, c.smoke, B)
    noopen = {"combo": {"id": cid + "-noopen", "sources": list(chosen["sources"]), "pool": POOL,
                        "calibrated": True, "opening": False},
              "params": params_dict(theta0, False, B), "refit": refit0,
              "oof_rank_balanced": m0["rank_balanced"], "oof_question_mean": m0["question_mean"],
              "oof_per_level": m0["per_level"], "folds": folds0}
    r10 = load_r10(c)
    if r10 is None:
        use = "noopen"
        why = "no fallback_R10.json given (smoke): the no-opening version"
    elif m0["rank_balanced"] > r10["inner_oof_rank_balanced"] + FLOAT_EPS:
        use, why = "noopen", "its inner OOF is higher than R10's"
    else:
        use, why = "R10", "R10's inner OOF is not lower (ties keep the registered R10)"
    fail_opening = {"use": use, "reason": why, "noopen_inner_oof_rank_balanced": m0["rank_balanced"],
                    "R10": r10, "rule": "PREREG_03 2.3: on test day T1 = FAIL_OPENING use the higher inner OOF of "
                                        "the selected combination's no-opening version and R10 (fixed now)"}
    if cid == "P00":
        fail_opening["P00_noopen_equals_R10_oof"] = (None if r10 is None else
                                                     abs(m0["rank_balanced"] - r10["inner_oof_rank_balanced"]))
    single_report, single_frozen = single_colour_tune_records(ms, mono_records, mono_scores, mono_boot, cid, cr)
    # PREREG_03 rev 3 11(c): a selected C6 source that cannot be used on test day blocks H3R (scoring chance kept)
    single_block = single_colour_score_block(cid, single_frozen.get("kappa_reference"))
    score_allowed = bool(gates["passed"]) and cid != "P00" and not single_block
    used = combo_dumps(chosen["sources"])
    op = inner["op"]
    inputs = dict(inner["inputs"])
    inputs.update({f"inner_{n}.npz": d["sha256"] for n, d in loaded.items()})
    inputs["check_report.json"] = sha256_file(os.path.join(c.out_dir, "check_report.json"))
    inputs["selfcheck.json"] = sha256_file(os.path.join(c.out_dir, "selfcheck.json"))
    report = {
        "what": "PREREG_03 competition 2 tune: P00..P06 on inner pseudo-questions, 5-fold OOF, gates G1-G3",
        "created": now_iso(), "seconds": round(time.time() - t0, 1), "smoke": c.smoke,
        "definitions": {"sources": SOURCE_TEXT,
                        "combination": "orientation lists of the sources concatenated; l = log_softmax(mean of "
                                       "the log_softmax arrays) (equal weight per source)",
                        "formula": "p = softmax(e^a (n/12)^gamma (m + w m_op + b)), log pooling, b = B beta",
                        "objective": f"p02: weighted NLL + {PENALTY} ||b||^2, L-BFGS-B bounds {BOUNDS}, starts "
                                     f"a in {STARTS}",
                        "oof": "p02: fold f fits on fit questions (rounds 0-49) of the other folds' players, "
                               "predicts eval questions (rounds 1000-1199) of fold f",
                        "bootstrap": f"players within each level, {N_BOOT} draws, default_rng({BOOT_SEED}), paired "
                                     f"with P00"},
        "usable_sources": usable, "combinations": results, "selection": selection, "gates": gates,
        "score_allowed": score_allowed, "score_blocked_single_colour": single_block,
        "se_val": se, "per_level_diff_vs_P00": per_level_diff,
        "refit": dict(refit, combo=cid), "no_opening": noopen, "fail_opening": fail_opening,
        "single_colour": single_report,
        "questions": {"fit": len(inner["X_fit"]), "eval": len(X_eval)},
        "inputs_sha256": inputs, "p02_inputs": p02, "arms_commit_check": commit, "order_check": order,
        "tool": tools, "frozen_tools": frozen, "environment": environment()}
    frozen_obj = {
        "format": FROZEN_FORMAT,
        "what": "PREREG_03 competition 2 frozen configuration (selected on inner; commit before apply)",
        "created": report["created"], "smoke": c.smoke,
        "combo": {"id": cid, "sources": list(chosen["sources"]), "pool": POOL, "calibrated": True, "opening": True},
        "source_definitions": {s: SOURCE_TEXT[s] for s in chosen["sources"]},
        "dumps": {n: {"weights_sha256": loaded[n]["weights_sha256"], "syms": N_SYMS,
                      "inner_sha256": loaded[n]["sha256"],
                      "folder": "prereg02 dumps" if n == C5_DUMP else f"prereg03 rank_{n[3]} dumps"} for n in used},
        "params": final, "basis": [[float(x) for x in row] for row in B], "formula": report["definitions"]["formula"],
        "decision_rule": "B, tools/decide.py (argmax p_k + e^-1 (p_{k-1} + p_{k+1}))",
        "opening_build_id": op["meta"]["build_id"],
        "selection": selection, "gates": gates, "score_allowed": score_allowed,
        "score_blocked_single_colour": single_block, "se_val": se,
        "oof": {k: chosen["oof"][k] for k in ("rank_balanced", "question_mean", "per_level", "per_n_bin",
                                             "pred_distribution")},
        "P00_oof_rank_balanced": p00["oof"]["rank_balanced"], "per_level_diff_vs_P00": per_level_diff,
        "no_opening": {k: noopen[k] for k in ("combo", "params", "oof_rank_balanced", "oof_question_mean")},
        "fail_opening": fail_opening,
        "single_colour": single_frozen,
        "check": {"usable_sources": usable, "check_report_sha256": inputs["check_report.json"],
                  "selfcheck_sha256": inputs["selfcheck.json"]},
        "inputs_sha256": inputs, "tool": tools, "environment": environment()}
    os.makedirs(c.out_dir, exist_ok=True)
    write_json(os.path.join(c.out_dir, "tune_report.json"), report)
    write_json(frozen_path, frozen_obj)
    print(f"selected {cid} (OOF rank-balanced {chosen['oof']['rank_balanced']:.4f}; best {selection['best_id']} "
          f"{selection['best_rank_balanced']:.4f}; P00 {p00['oof']['rank_balanced']:.4f}); gates "
          f"{'PASSED' if gates['passed'] else 'not passed'} ({gates['reason']}); score allowed {score_allowed}")
    for why_blocked in single_block:
        print(f"score blocked: {why_blocked}")
    if se:
        print(f"recorded: SE_val {se['se_val']:.4f}, adoption probability {se['adoption_probability']}")
    print(f"FAIL_OPENING fallback: {use} ({why}); no-opening OOF {m0['rank_balanced']:.4f}")
    if single_frozen.get("status") == "ok":
        s_sel, s_p00 = single_frozen["selected"], single_frozen["P00"]
        print(f"single-colour inner (record only): {cid} {s_sel['rank_balanced']:.4f} (B "
              f"{s_sel['by_colour']['B']['rank_balanced']}, W {s_sel['by_colour']['W']['rank_balanced']}); P00 "
              f"{s_p00['rank_balanced']:.4f} (B {s_p00['by_colour']['B']['rank_balanced']}, W "
              f"{s_p00['by_colour']['W']['rank_balanced']})")
    print(f"wrote tune_report.json and frozen_config.json in {c.out_dir}; commit frozen_config.json before apply")
    return 0


# ================================================================ apply
def cmd_apply(a):
    t0 = time.time()
    c = resolve(a, "apply")
    fpath = os.path.join(c.out_dir, "frozen_config.json")
    commit = None
    if not c.smoke:
        commit = require_tracked_clean([fpath], "frozen_config.json (commit it before apply)")
        rc, _ = git(["diff", "--quiet", "HEAD", "--", repo_rel(fpath)])
        if rc != 0:
            raise P03Error("frozen_config.json differs from HEAD; apply only runs on the committed frozen config")
    elif not os.path.exists(fpath):
        raise P03Error("run tune first (frozen_config.json missing)")
    frozen = start(c)
    fc = read_json(fpath)
    if fc.get("format") != FROZEN_FORMAT:
        raise P03Error(f"{fpath}: not a {FROZEN_FORMAT} file")
    if bool(fc.get("smoke")) != c.smoke:
        raise P03Error("frozen_config.json smoke flag differs from this run")
    B = np.asarray(fc["basis"], dtype=np.float64)
    if B.shape != (N_LEVELS, N_LEVELS - 1) or not np.array_equal(B, basis()):
        raise P03Error("frozen basis differs from the fixed basis")
    tools = tool_state()
    if not c.smoke:
        changed = [f for f, h in tools["files_sha256"].items() if fc["tool"]["files_sha256"].get(f) != h]
        if changed:
            raise P03Error(f"tool files changed since tune: {changed}; rerun tune (and revise) first")
    combo = fc["combo"]
    srcs = tuple(combo["sources"])
    if (combo["id"], srcs) not in COMBOS or combo.get("pool") != POOL or combo.get("opening") is not True:
        raise P03Error(f"frozen combination {combo} is not one of P00..P06 (log pooling, opening)")
    arms_used = [SOURCE_ARM[s] for s in srcs if s in SOURCE_ARM]
    results_commit = arms_committed(c, arms_used)
    p02 = p02_inputs(c, val=True)
    par = fc["params"]
    theta = theta_of(par, True)
    if not np.allclose(B @ np.asarray(par["beta"]), par["b"], atol=1e-12):
        raise P03Error("frozen b is not B @ beta")
    vk, logits, dump_sha, wsha = load_val_side(c, srcs, c.arm_dir)
    for n, info in fc["dumps"].items():
        if wsha.get(n) != info["weights_sha256"]:
            raise P03Error(f"val_{n}.npz weights {wsha.get(n)} differ from the tuned inner dump's "
                           f"{info['weights_sha256']}")
    crp = os.path.join(c.out_dir, "check_report.json")
    if os.path.exists(crp):
        cr = read_json(crp)
        for s in srcs:
            if s == "C5":
                continue
            rec = (cr["sources"].get(s) or {}).get("inputs_sha256") or {}
            n = SOURCES[s][0][0]
            if rec.get(f"val_{n}.npz") != dump_sha[f"val_{n}.npz"]:
                raise P03Error(f"val_{n}.npz changed since check")
    elif not c.smoke:
        raise P03Error("check_report.json missing")
    questions = read_question_meta(c.questions)
    by_q = {q: (n, cols) for q, n, cols in questions}
    groups = vk["groups"]
    qids = [q for q, _ in groups]
    if c.smoke:
        if qids != [q for q, _, _ in questions][:len(qids)]:
            raise P03Error("smoke val dump questions are not the first questions of the question file")
    elif qids != [q for q, _, _ in questions]:
        raise P03Error("val_keys questions differ from the question file (set or order)")
    for q, rows in groups:
        n, cols = by_q[q]
        if len(rows) != n or [vk["color"][i] for i in rows] != cols:
            raise P03Error(f"question {q}: val_keys slots/colours differ from the question file")
    op_o, op_has, op = opening_val_aligned(c, vk, fc["opening_build_id"])
    P, n, ok = val_probs(srcs, theta, True, B, vk, logits, op_o, op_has)
    os.makedirs(c.out_dir, exist_ok=True)
    out_csv = os.path.join(c.out_dir, "rank_val_probs.csv")
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(PROB_COLUMNS)
        for k, (q, rows) in enumerate(groups):
            probs = [repr(float(x)) for x in P[k]] if ok[k] else [""] * N_LEVELS
            w.writerow([q, len(rows), int(n[k])] + probs)
    os.replace(tmp, out_csv)
    decide.read_probs(out_csv)   # must pass decide.py's strict reader
    gates = fc.get("gates")
    blocked = []
    if not isinstance(gates, dict) or gates.get("passed") is not True:
        blocked.append("gates not passed in tune (PREREG_03 2.4): do not score; record 'not submitted'")
    if combo["id"] == "P00":
        blocked.append("P00 (R20) selected: nothing new to score")
    # PREREG_03 rev 3 11(c): recomputed from the frozen single-colour kappa reference of the selected sources
    single_block = single_colour_score_block(combo["id"], (fc.get("single_colour") or {}).get("kappa_reference"))
    blocked.extend(single_block)
    rep = {"what": "PREREG_03 competition 2 apply: frozen configuration on the validation questions",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), "smoke": c.smoke,
           "commit_check": commit, "arms_commit_check": results_commit, "combo": combo, "params": par,
           "questions": len(groups), "questions_without_parsed_game": int((~ok).sum()),
           "games": vk["n"], "games_parsed": int(vk["parsed"].sum()),
           "inputs_sha256": dict(dump_sha, **{"val_keys.npz": vk["sha256"], "frozen_config.json": sha256_file(fpath),
                                              "questions": sha256_file(c.questions),
                                              "opening_val.npz": op["sha256"]}),
           "p02_inputs": p02, "output": os.path.abspath(out_csv), "output_sha256": sha256_file(out_csv),
           "tool": tools, "frozen_tools": frozen, "environment": environment(),
           "gates": gates, "score_allowed": not blocked, "score_blocked_reasons": blocked,
           "score_blocked_single_colour": single_block,
           "next": "PY tools/decide.py <out-dir>/rank_val_probs.csv <out-dir>; PY tools/check_submission.py rank "
                   "<out-dir>/pred_B.csv --questions <question file>; commit; score (PREREG_03 3) only when "
                   "score_allowed is true"}
    write_json(os.path.join(c.out_dir, "apply_report.json"), rep)
    print(f"apply {combo['id']}: {len(groups)} questions ({int((~ok).sum())} without a parsed game) -> {out_csv}")
    print("score allowed: %s%s" % (not blocked, (" (" + "; ".join(blocked) + ")") if blocked else ""))
    print(f"next: PY tools/decide.py {out_csv} {c.out_dir}")
    return 0


# ================================================================ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["unpack", "check", "selfcheck", "tune", "apply"])
    ap.add_argument("zip", nargs="?", help="unpack: the arm's result zip")
    ap.add_argument("--arm", choices=ARMS, help="unpack: which arm")
    ap.add_argument("--dest", help="unpack --smoke: target folder outside the repository")
    ap.add_argument("--p02-dir", help="prereg02 rank result dir with dumps/ (real: results/prereg02/rank)")
    ap.add_argument("--p02-local-dir", help="pseudo_inner.json, opening_*.npz and R20's tune/frozen/probs files "
                                            "(real: results/prereg02/rank)")
    ap.add_argument("--p03-dir", help="folder holding rank_a/ and rank_b/ (real: results/prereg03)")
    ap.add_argument("--out-dir", help="real: results/prereg03/rank (or outside the repository for a preview of "
                                      "check / selfcheck); --smoke: required, outside the repository")
    ap.add_argument("--split", help="players_split.csv (real: results/split_v1/players_split.csv)")
    ap.add_argument("--fallback-r10", help="fallback_R10.json (real: results/test02/rank/fallback_R10.json)")
    ap.add_argument("--questions", help="apply: question file (real: data/local/rank_val_questions.csv)")
    ap.add_argument("--min-acc", type=float, help="--smoke only: floor in percent (default 24.15)")
    ap.add_argument("--smoke-record-colour", action="store_true",
                    help="--smoke only (check): record the colour wiring / kappa stops instead of stopping (an "
                         "untrained SMOKE model's colour signal is noise)")
    ap.add_argument("--smoke", action="store_true", help="SMOKE / synthetic inputs; output outside the repository")
    a = ap.parse_args(argv)
    try:
        if a.command == "unpack":
            if not a.zip or not a.arm:
                raise P03Error("unpack needs --arm a|b and the zip path")
            return cmd_unpack(a)
        if a.zip or a.arm or a.dest:
            raise P03Error("the zip path, --arm and --dest belong to unpack only")
        return {"check": cmd_check, "selfcheck": cmd_selfcheck, "tune": cmd_tune, "apply": cmd_apply}[a.command](a)
    except ERRORS as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
