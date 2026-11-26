"""PREREG_19 competition 2: freeze Q19 = Q09 + 4 x SET (the question-level set model S19 as a 7th fusion term) and
check it on the validation questions without labels (docs/PREREG_19_SET_MODEL.md rev 1, section 2).

    PY = .venv/Scripts/python.exe   (CPU; Git Bash; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL thread variable;
                                     from a path without 'AppData\\Local' (u1_lib's data/local guard), e.g. a subst drive)
    PY tools/p19_set.py freeze                  -> results/prereg19/rank/s19_seed{0,1,2}.npz (copies), freeze_report.json,
                                                   frozen_config.json (run once; commit them before apply)
    PY tools/p19_set.py apply                   -> rank_val_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json,
                                                   check_submission.txt, val_set.npz, apply_report.json (no score)
    PY tools/p19_set.py apply --finalize        (after apply_audit.json is committed: binds it, sets usable)

freeze (PREREG_19 2.1): the inner evaluation of the exploration (tools/p09_rank_fuse.Inner on the committed inner
    P1 features data/prereg12/q09_inner), seams 1 (Q02H / Q08 / Q09 refs), M (model files, exploration records), E
    (own pooling = Inner.E_p1), F (Q19 OOF = eval_report.json's candidate), then ONE fit of the 7-term theta on all
    inner fit pseudo-questions (p04_rank_fuse.gen_fit_record) and the direction references.
apply (PREREG_19 2.2): Q02, Q02H, Q08 and Q09 reproduced byte for byte on the validation questions (the registered
    readers; data/local/rank_val_questions.csv is read only through p07_rank_fuse.q02_validation / check_question_file
    and check_submission --questions: question ids, counts, colours, SGF), lp_SET from 08's validation chunks, Q19 with
    the frozen theta, decide.py, check_submission, the direction check; usable 'pending_audit' until --finalize.

Discipline: never imports torch; u1_lib only inside freeze (through p09_rank_fuse.Inner); apply never imports u1_lib
and reads data/local before pilot_fuse / p1_lib are imported; the validation label file is never named or opened;
tools/score.py is never run; the ledger gets no row. Exit codes: 0 done; 1 the step ran and says stop (a seam
failed); 2 refused or error.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import p09_rank_fuse as r9  # noqa: E402  (pins the p08 tools before importing them)
import p09_forward as FW  # noqa: E402
import p19_forward as SF  # noqa: E402

R, rf, p3, p4, P7 = r9.R, r9.rf, r9.p3, r9.p4, r9.P7


class P19Error(RuntimeError):
    pass


class Stop(RuntimeError):
    """The step ran; its registered outcome is a stop (exit 1)."""


ERRORS = (P19Error, SF.SetForwardError) + r9.ERRORS

# ================================================================ registered values (PREREG_19 rev 1)
L = rf.N_LEVELS
LEVELS = rf.LEVELS
TOL = 1e-12
SCALE = SF.SCALE
TERMS = list(r9.Q08_TERMS) + ["SET"]                 # OP, O4, SWA, SWB, HS, P1TO (= P1TO_E), SET
RHO_FLOOR_VAL = r9.RHO_FLOOR_VAL                      # 0.078
RHO_FLOOR_TEST = r9.RHO_FLOOR_TEST                    # 0.155
EXPLORE = os.path.join(REPO, "results", "prereg19", "explore")
EXPLORE_PINS = {  # LF SHA-256 at 85cbe04 (PREREG_19 1.1: specification and the judged records; never imported)
    "results/prereg19/explore/PLAN.md": "6653f332976030f1527a5f7df8de59a1840e6f8a69c0fb62742e2f92a4f25eba",
    "results/prereg19/explore/p19_lib.py": "bbdc4225687777b732df1275277c4f4986f77225b935808f02cc7aefd4256376",
    "results/prereg19/explore/p19_train.py": "9493ef1367b01495769a40ab792dc2b57208a9555de25a8f118147501b7b10ed",
    "results/prereg19/explore/p19_eval.py": "1400861cb9e097c3110776cd50fae81e41fa409af8588b66e685bd4c5861ea8a",
    "results/prereg19/explore/train_record.json": "89b0c11c25ca61e0381c7e22e0869de06e403222358a0a1a4655b736890d770b"}
EXPLORE_RECORDS = ("results/prereg19/explore/eval_report.json", "results/prereg19/explore/audit.json")
EXPLORE_COMMIT = "85cbe04"
# the exploration's judged candidate (eval_report.json, 17f1974): seam F
REF_Q19 = {"inner": 0.7345843051712626, "mono": 0.7430172362615272}
REF_Q09 = {"inner": 0.7114420614163589, "mono": 0.7145431684624712}
P09_RANK = os.path.join(REPO, "results", "prereg09", "rank")
P09_PINS = {  # the committed prereg09 files 19 builds on (pin_sha: .json / .csv LF, .npz raw)
    "results/prereg09/rank/frozen_config.json": "50d38d2787c94569d88ed87a482e872ed64be2d1d20e4f57ffbb07a216663ff5",
    "results/prereg09/rank/p1e_map.npz": "ecc9c9b1747b7b7d40cf53869b6c67c8e6f776ba7498498737fe3eede8e2edba",
    "results/prereg09/rank/rank_val_probs.csv": "3f8d32f4b5b0431c9f4407199ae2c074595428ef9053ab807a5cf3b129e3b4fb",
    "results/prereg09/rank/pred_B.csv": "d1a2f1d673a8b95425dfdc6b392a64a3e7484509d8bbaa3eb3c72f5ee283333b",
    "results/prereg09/rank/val_lp_p1e.npz": "dc70be8a0f57dddb51836f8fb47ab598ac02ffaac6803949fdf4770004463cdf",
    "results/prereg09/rank/apply_report.json": "691cd3a7c06e86947d96af1d2faf152c52006e336ebe15e81124ebeb3fd773b3",
    "results/prereg09/rank/tune_report.json": "30fe15a0bfec2160dfcb4a50e3a7f5ad7c7de13f3e8a35c341b9912d66abba83",
    "results/prereg09/rank/check_report.json": "3dfe765d1b7aa1c7c9f33ac8c0d6f89e44a5abf0141ca1195fdaa0fc71fe00a0"}
INNER_PREP = os.path.join(REPO, "data", "prereg12", "q09_inner")
INNER_FILES = ("inner_X.npy", "inner_ex.npy", "inner_meta.npz")
MODELS_SRC = os.path.join(REPO, "data", "prereg19", "models")
OUT = os.path.join(REPO, "results", "prereg19", "rank")
PREREG = os.path.join(REPO, "docs", "PREREG_19_SET_MODEL.md")
PREREG_REL = "docs/PREREG_19_SET_MODEL.md"
TOOL_FILES = ("tools/p19_set.py", "tools/p19_forward.py", "tools/p09_rank_fuse.py", "tools/p09_forward.py",
              "tools/p08_rank_fuse.py", "tools/p07_rank_fuse.py", "tools/p04_rank_fuse.py", "tools/p03_rank_fuse.py",
              "tools/p02_rank_fuse.py", "tools/decide.py", "tools/check_submission.py")
FORMATS = {"freeze": "c10-prereg19 rank freeze_report v1", "frozen": "c10-prereg19 rank frozen_config v1",
           "apply": "c10-prereg19 rank apply_report v1", "audit": "c10-prereg19 apply_audit v1",
           "val_set": "c10-prereg19 val_set v1"}
APPLY_OUT = ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "check_submission.txt",
             "val_set.npz", "apply_report.json")
AUDIT_BOUND_OUTPUTS = ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "val_set.npz")
AUDIT_BOUND_INPUTS = ("frozen_config.json", "s19_seed0.npz", "s19_seed1.npz", "s19_seed2.npz", "val_report.json (08)")
MIN_REV_RERUN = 2                                     # a rerun of a written step cites PREREG_19 rev 2 or later
_LABEL = "ans" + "wer"

path_of, rel, git, commit_time, tracked_clean = P7.path_of, P7.rel, P7.git, P7.commit_time, P7.tracked_clean
sha_raw, sha_lf, pin_sha, read_json, write_json = r9.sha_raw, r9.sha_lf, r9.pin_sha, r9.read_json, r9.write_json
jsonable = P7.jsonable


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(*x):
    print(*x, flush=True)


def refuse_label_path(path):
    if path is not None and _LABEL in os.path.abspath(os.fspath(path)).lower():
        raise P19Error(f"refusing {path}: the validation label file is never opened by the prereg19 tools")
    return path


def git_blob(ref, rel_):
    """The bytes of rel_ at ref (None when absent): no decoding, no strip."""
    try:
        r = subprocess.run(["git", "-C", REPO, "show", f"{ref}:{rel_}"], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def show(path):
    """A repository-relative name for messages and records (no realpath: data/ is a junction in a worktree)."""
    try:
        return os.path.relpath(os.path.abspath(path), REPO).replace(os.sep, "/")
    except ValueError:
        return os.path.abspath(path)


def tool_state():
    rc, st = git(["status", "--porcelain", "--untracked-files=all", "--"] + list(TOOL_FILES))
    dirty = [ln[3:] for ln in st.splitlines() if ln.strip()] if rc == 0 else None
    return {"git_head": P7.git_head(), "dirty_or_untracked": dirty, "files_sha256_rule": "sha256, CRLF folded to LF",
            "files_sha256": {f: (sha_lf(path_of(f)) if os.path.exists(path_of(f)) else None) for f in TOOL_FILES}}


def require_clean_tools():
    ts = tool_state()
    if ts["git_head"] is None or ts["dirty_or_untracked"] is None:
        raise P19Error("git is not available")
    if ts["dirty_or_untracked"]:
        raise P19Error(f"commit the tools first (uncommitted: {ts['dirty_or_untracked']})")
    if tracked_clean([PREREG]):
        raise P19Error("docs/PREREG_19_SET_MODEL.md must be committed and clean")
    return ts


def venv_record():
    """python / numpy / scipy of this process (the test day compares them with frozen_config.json)."""
    return r9.venv_versions(False)


def check_pins():
    """The exploration's records and code (85cbe04) and the prereg09 files 19 builds on keep their SHA-256s."""
    got, bad = {}, []
    for r_, want in EXPLORE_PINS.items():
        got[r_] = sha_lf(path_of(r_)) if os.path.exists(path_of(r_)) else None
        if got[r_] != want:
            bad.append(r_)
    for r_ in EXPLORE_RECORDS:
        blob = git_blob(EXPLORE_COMMIT, r_)
        want = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() if blob is not None else None
        got[r_] = sha_lf(path_of(r_)) if os.path.exists(path_of(r_)) else None
        if want is None or got[r_] != want:
            bad.append(r_)
    for r_, want in P09_PINS.items():
        got[r_] = pin_sha(path_of(r_)) if os.path.exists(path_of(r_)) else None
        if got[r_] != want:
            bad.append(r_)
    if bad:
        raise P19Error(f"pinned files changed: {bad} (PREREG_19 1.1: never edited)")
    return got


def revision_after(n, when):
    """PREREG_19 rev N (N >= MIN_REV_RERUN) at HEAD of the committed, clean registration, added after `when` (ISO)."""
    if n is None or n < MIN_REV_RERUN:
        raise P19Error(f"a rerun needs --revision N (PREREG_19 rev {MIN_REV_RERUN} or later naming the program error)")
    if tracked_clean([PREREG]):
        raise P19Error("docs/PREREG_19_SET_MODEL.md must be committed and clean")
    row = f"| rev {n} |"
    head = git_blob("HEAD", PREREG_REL)
    if head is None or not any(x.startswith(row) for x in head.decode("utf-8", "replace").splitlines()):
        raise P19Error(f"{PREREG_REL} at HEAD has no row starting {row!r}")
    rc, out = git(["log", "--reverse", "-S" + row, "--format=%H %cI", "--", PREREG_REL])
    if rc != 0 or not out:
        raise P19Error(f"no commit added {row!r}")
    h, t = out.splitlines()[0].split()
    if when and datetime.datetime.fromisoformat(t) <= datetime.datetime.fromisoformat(when):
        raise P19Error(f"PREREG_19 rev {n} ({t}) is not later than the record it replaces ({when})")
    return {"rev": n, "commit": h, "committed_at": t}


# ================================================================ the models
def model_paths(folder=OUT):
    return [os.path.join(folder, SF.model_name(s)) for s in SF.SEEDS]


def install_models():
    """PREREG_19 1.2: the three exploration model files copied byte for byte into results/prereg19/rank (once)."""
    rec = {}
    for s, dst in zip(SF.SEEDS, model_paths()):
        src = os.path.join(MODELS_SRC, SF.model_name(s))
        if not os.path.exists(dst):
            if not os.path.exists(src) or SF.sha256_file(src) != SF.MODEL_SHA256[s]:
                raise P19Error(f"{show(src)} is missing or not the registered model of seed {s}")
            os.makedirs(OUT, exist_ok=True)
            shutil.copyfile(src, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
        got = SF.sha256_file(dst)
        src_sha = SF.sha256_file(src) if os.path.exists(src) else None
        if got != SF.MODEL_SHA256[s]:
            raise P19Error(f"{rel(dst)} is not the registered model of seed {s}")
        rec[str(s)] = {"file": show(dst), "sha256": got, "source": show(src), "source_sha256": src_sha,
                       "best_step": SF.BEST_STEP[s]}
    tr = read_json(path_of("results/prereg19/explore/train_record.json"))
    for s in SF.SEEDS:
        if tr["models_sha256"][str(s)] != SF.MODEL_SHA256[s] or tr["seeds"][str(s)]["best_step"] != SF.BEST_STEP[s]:
            raise P19Error(f"train_record.json seed {s} differs from PREREG_19 1.2")
    return rec


# ================================================================ freeze (PREREG_19 2.1)
class InnerPrep:
    """The committed inner P1 features (data/prereg12/q09_inner; = ft_eval.InnerPrep without importing p09_map)."""

    def __init__(self, d):
        self.dir = d
        self.inner_X = np.load(os.path.join(d, "inner_X.npy"), mmap_mode="r")
        self.inner_ex = np.load(os.path.join(d, "inner_ex.npy"), mmap_mode="r")
        with np.load(os.path.join(d, "inner_meta.npz"), allow_pickle=False) as z:
            self.inner_ref = np.asarray(z["ref"]).astype(np.int64)
            self.inner_src = np.asarray(z["src"]).astype(str)
        if len(self.inner_ref) != r9.INNER_USABLE or len(self.inner_X) != r9.INNER_USABLE:
            raise P19Error(f"inner rows {len(self.inner_ref)} != {r9.INNER_USABLE}")

    def inner_rows(self, sets=("bank", "mono")):
        return np.nonzero(np.isin(self.inner_src, np.array(list(sets))))[0]


def inner_prep_seam():
    """Seam P: inner_X.npy, inner_ex.npy and inner_meta.npz equal the files prereg09's check_report.json bound (the
    features of the exploration and of Q09's tune)."""
    want = (read_json(path_of("results/prereg09/rank/check_report.json")).get("prep") or {}).get("files_sha256") or {}
    got = {n: sha_raw(os.path.join(INNER_PREP, n)) for n in INNER_FILES}
    return {"got": got, "want": {n: want.get(n) for n in INNER_FILES},
            "ok": bool(all(want.get(n) and got[n] == want.get(n) for n in INNER_FILES))}


def inner_set_terms(prep, inner, par_list, mu_x, sd_x):
    """E_SET (x 4) on the fit / eval / mono pseudo-questions; usable = Inner.covered (= Q09's P1TO mask); h on the
    prepared rows in blocks of 8,192 (as the exploration)."""
    Hc = SF.game_h(par_list, prep.inner_X, mu_x, sd_x)
    H = []
    for hc in Hc:
        h = np.zeros((inner.n_inner, SF.H_DIM))
        h[prep.inner_ref] = hc
        H.append(h)
    del Hc
    di = inner.d["inner"]
    E, info = {}, {}
    for which, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"])):
        groups = SF.usable_groups(S, inner.covered)
        E[which], _per, _lp, has = SF.set_term(par_list, H, groups, inner.n_inner)
        info[which] = {"questions": int(len(groups)), "without_usable_game": int((~has).sum())}
    return E, info


def seam_e(inner, lp_q09):
    E_ref, _ = inner.E_p1(lp_q09)
    lp4 = SCALE * inner.lp_full(lp_q09)
    di = inner.d["inner"]
    out = {}
    for which, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"])):
        g = SF.usable_groups(S, inner.covered)
        out[which] = float(np.max(np.abs(np.asarray(SF.mean_matrix(g, inner.n_inner) @ lp4) - E_ref[which])))
    out["ok"] = bool(all(v <= TOL for v in out.values()))
    return out


def cmd_freeze(a):
    t0 = time.time()
    r9.blas_env_check(False)
    venv = venv_record()
    fc9 = read_json(path_of("results/prereg09/rank/frozen_config.json"))
    want_v = {k: ((fc9.get("environment") or {}).get("versions") or {}).get(k) for k in ("python", "numpy", "scipy")}
    if {k: venv[k] for k in want_v} != want_v:
        raise P19Error(f".venv {venv} differs from prereg09's frozen versions {want_v}")
    frozen_path, rep_path = os.path.join(OUT, "frozen_config.json"), os.path.join(OUT, "freeze_report.json")
    rerun = None
    if os.path.exists(frozen_path) or os.path.exists(rep_path):
        if not (a.rerun and a.reason):
            raise P19Error("freeze ran already (frozen_config.json / freeze_report.json exist): run once; a rerun "
                           "needs --rerun --reason TEXT --revision N")
        prior = [p for p in (frozen_path, rep_path) if os.path.exists(p)]
        when = max((commit_time(p) or "") for p in prior) or None
        rerun = dict(revision_after(a.revision, when), reason=a.reason,
                     replaced={rel(p): pin_sha(p) for p in prior})
    ts = require_clean_tools()
    pins = check_pins()
    models = install_models()
    par_list = SF.load_models(OUT)
    seam_p = inner_prep_seam()
    if not seam_p["ok"]:
        raise P19Error(f"seam P: {seam_p}")
    # ---- the inner evaluation (u1_lib through p09_rank_fuse.Inner: training players only)
    prep = InnerPrep(INNER_PREP)
    inner = r9.Inner(prep, log=log)
    mp = FW.load_map(path_of("results/prereg09/rank/p1e_map.npz"))
    lp_q09 = r9.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    q09 = inner.run(list(r9.Q08_TERMS), {"HS": inner.E_hs, "P1TO": E_q09})
    seam1 = {"Q02H": [inner.h["rb"], r9.REF["Q02H"]], "Q08": [inner.q08["rb"], r9.REF["Q08"]],
             "Q08_mono": [inner.q08["mono_rb"], r9.REF["Q08_mono"]], "Q09": [q09["rb"], REF_Q09["inner"]],
             "Q09_mono": [q09["mono_rb"], REF_Q09["mono"]]}
    seam1_ok = all(abs(x - y) <= TOL for x, y in seam1.values())
    se = seam_e(inner, lp_q09)
    log(f"seam 1 {seam1_ok} {seam1}; seam E {se}")
    if not (seam1_ok and se["ok"]):
        raise Stop(f"seam 1 / E failed: {seam1} {se}")
    # ---- the SET term and seam F (the exploration's judged candidate, reproduced)
    tc = time.time()
    E_set, set_info = inner_set_terms(prep, inner, par_list, mp["mu_x"], mp["sd_x"])
    cand = inner.run(TERMS, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_set})
    ev = read_json(path_of("results/prereg19/explore/eval_report.json"))
    ev_w = (ev.get("candidate") or {}).get("fusion_weights_by_fold") or []
    my_w = [fr["params"]["w"] for fr in cand["folds"]]
    w_diff = max((abs(x[k] - y[k]) for x, y in zip(my_w, ev_w) for k in TERMS), default=None) \
        if len(my_w) == len(ev_w) and all(set(x) == set(TERMS) for x in ev_w) else None
    seamf = {"inner": [cand["rb"], REF_Q19["inner"]], "mono": [cand["mono_rb"], REF_Q19["mono"]],
             "eval_report_inner": (ev.get("candidate") or {}).get("inner_rank_balanced"),
             "eval_report_mono": (ev.get("candidate") or {}).get("single_colour_rank_balanced"),
             "fold_weights_max_abs_diff": w_diff}
    seamf["ok"] = bool(abs(cand["rb"] - REF_Q19["inner"]) <= TOL and abs(cand["mono_rb"] - REF_Q19["mono"]) <= TOL
                       and seamf["eval_report_inner"] == REF_Q19["inner"]
                       and seamf["eval_report_mono"] == REF_Q19["mono"] and w_diff is not None and w_diff <= TOL)
    log(f"SET term {set_info} ({time.time() - tc:.0f}s); seam F {seamf}")
    if not seamf["ok"]:
        raise Stop(f"seam F failed (the tool does not reproduce the exploration's candidate): {seamf}")
    cmp_ = inner.compare(cand, q09)
    # ---- ONE fit of the 7-term theta on all inner fit pseudo-questions (PREREG_19 1.3)
    theta, refit = p4.gen_fit_record(cand["aux"]["mf"], cand["aux"]["Ef"], inner.Xf.n, inner.Xf.y)
    params = p4.gen_params(theta, TERMS)
    Pfit = R.fuse_probs(theta, cand["aux"]["mf"], cand["aux"]["Ef"], inner.Xf.n, np.ones(len(inner.Xf), dtype=bool))
    mfit, _ = rf.metrics(rf.rule_b_np(Pfit), inner.Xf.y, inner.Xf.n, inner.Xf.player)
    refit.update(fit_questions=len(inner.Xf), bounds_hit=P7.bound_hits(params["w"]),
                 in_sample_fit_rank_balanced_record_only=mfit["rank_balanced"],
                 in_sample_fit_question_mean_record_only=mfit["question_mean"])
    dref = r9.direction_ref(inner, E_set, inner.q02)
    dref["definition"] = dref["definition"].replace("E_P1 = masked mean of 4 lp of the candidate map",
                                                    "E_SET = 4 lp_SET of the question's usable games")
    folds = [{"fold": fr["fold"], "loss": fr["loss"], "params": fr["params"],
              "bounds_hit": P7.bound_hits(fr["params"]["w"])} for fr in cand["folds"]]
    common = {"created": now_iso(), "tool": ts, "venv": venv, "pinned_sha256": pins, "models": models,
              "inner_prep": {"dir": show(INNER_PREP), **seam_p}, "rerun": rerun}
    report = {"format": FORMATS["freeze"], "what": "PREREG_19 2.1 freeze: seams 1 / M / P / E / F, then one fit of the "
                                                   "7-term theta on all inner fit pseudo-questions",
              **common, "seconds": round(time.time() - t0, 1), "seam_1": seam1, "seam_e": se, "seam_f": seamf,
              "set_term": set_info, "oof_vs_q09_record": cmp_,
              "oof": {"Q19": {"inner_rank_balanced": cand["rb"], "single_colour_rank_balanced": cand["mono_rb"],
                              "folds": folds},
                      "Q09": {"inner_rank_balanced": q09["rb"], "single_colour_rank_balanced": q09["mono_rb"]}},
              "refit": refit, "params": params, "direction_ref": dref}
    frozen = {"format": FORMATS["frozen"],
              "what": "PREREG_19 competition 2 frozen configuration Q19 = Q09 + 4 x SET (inner only; adopted on the "
                      "exploration's single judgement, results/prereg19/explore/eval_report.json)",
              **common, "adopted_on": {"eval_report": "results/prereg19/explore/eval_report.json",
                                       "eval_commit": "17f1974", "audit": "results/prereg19/explore/audit.json",
                                       "audit_commit": EXPLORE_COMMIT, "gates": ev.get("gates")},
              "combo": {"id": "Q19", "terms": TERMS, "path": "gen", "pool": "log", "calibrated": True, "opening": True,
                        "p1_scale": SCALE, "p1_term": "P1TO_E (ens3fam, results/prereg09/rank/p1e_map.npz)",
                        "set_scale": SCALE, "set_term": "SET (S19, 3 seeds, results/prereg19/rank/s19_seed*.npz)"},
              "params": params,
              "formula": "p = softmax(e^a (n/12)^gamma (m_C5 + w_OP m_OP + w_O4 E_O4 + w_SWA E_SWA + w_SWB E_SWB + "
                         "w_HS E_HS + w_P1 E_P1E + w_SET E_SET + B beta)); E_P1E = mean over the question's usable "
                         "games of 4 lp_P1TO_E (Q09); E_SET = 4 lp_SET, lp_SET = log(mean over the 3 seeds of "
                         "head(mean over the usable games of h(x))), 0 without a usable game (tools/p19_forward.py)",
              "decision_rule": "B, tools/decide.py (argmax p_k + e^-1 (p_{k-1} + p_{k+1}))",
              "p1e_map": {"file": "results/prereg09/rank/p1e_map.npz",
                          "sha256": P09_PINS["results/prereg09/rank/p1e_map.npz"]},
              "q09_frozen_sha256": P09_PINS["results/prereg09/rank/frozen_config.json"],
              "direction_ref": dref,
              "oof": report["oof"], "refit": {k: refit[k] for k in ("loss", "starts", "fit_questions", "bounds_hit")},
              "environment": {"versions": venv, "cpu": r9.environment_cpu()}}
    write_json(rep_path, report)
    write_json(frozen_path, frozen)
    w = params["w"]
    log(f"frozen Q19: w {json.dumps({k: round(v, 4) for k, v in w.items()})}, a {params['a']:.4f}, gamma "
        f"{params['gamma']:.4f}; rho_eval {dref['rho_eval']:.4f}, rho_mono {dref['rho_mono']:.4f}; "
        f"{time.time() - t0:.0f}s")
    log("commit results/prereg19/rank/ (frozen_config.json, freeze_report.json, s19_seed*.npz) before apply")
    return 0


# ================================================================ apply (PREREG_19 2.2)
def frozen_usable(fc):
    if fc.get("format") != FORMATS["frozen"] or (fc.get("combo") or {}).get("id") != "Q19" \
            or (fc.get("combo") or {}).get("terms") != TERMS:
        raise P19Error("frozen_config.json is not the frozen Q19 configuration")
    if len((fc.get("params") or {}).get("theta") or []) != 2 + len(TERMS) + L - 1:
        raise P19Error("frozen_config.json theta has the wrong length")
    return True


def apply_prerequisites():
    fpath = os.path.join(OUT, "frozen_config.json")
    if not os.path.exists(fpath):
        raise P19Error("frozen_config.json missing (run freeze and commit it first)")
    fc = read_json(fpath)
    frozen_usable(fc)
    files = [fpath, os.path.join(OUT, "freeze_report.json")] + model_paths()
    rec = {"frozen": r9.require_committed(files, "frozen_config.json / freeze_report.json / models (commit first)")}
    for s, p in zip(SF.SEEDS, model_paths()):
        if SF.sha256_file(p) != SF.MODEL_SHA256[s] or (fc["models"][str(s)] or {}).get("sha256") != SF.MODEL_SHA256[s]:
            raise P19Error(f"{rel(p)} is not the registered model")
    rec["pins"] = check_pins()
    ts = require_clean_tools()
    diff = [f for f in TOOL_FILES if (fc.get("tool") or {}).get("files_sha256", {}).get(f) != ts["files_sha256"][f]]
    rec["tool_differs_from_freeze_record_only"] = diff
    rec["tool"] = ts
    vr = R.read_json(r9.P08_VAL_REPORT)
    if vr.get("format") != R.FORMATS["val_report"] or vr.get("mode") != "real" or vr.get("status") != "ok":
        raise P19Error("results/prereg08/feat/val_report.json is not a real, ok 08 validation report")
    return fc, vr, rec


def apply_decision(same, fail_count, repro, direction, formats_ok, audit):
    reasons = []
    if not same.get("passed"):
        reasons.append(f"usable games differ from prereg09's val_lp_p1e.npz: {same}")
    if fail_count != 0:
        reasons.append(f"validation P1: {fail_count} failed games (registered 0)")
    for k, v in repro.items():
        if v is not True:
            reasons.append(f"{k} does not reproduce its committed rank_val_probs.csv byte for byte")
    if not direction.get("passed"):
        reasons.append(f"direction check: rho {direction.get('rho')} < {direction.get('threshold')}")
    if formats_ok is not True:
        reasons.append("check_submission did not pass for pred_A.csv and pred_B.csv")
    if audit == "bug":
        reasons.append("apply_audit.json found a program error affecting the predictions")
    if reasons:
        return False, reasons
    return "pending_audit", ["apply_audit.json of these outputs not committed yet (PREREG_19 2.3); then apply "
                             "--finalize"]


def q19_core(q, core, qc9, set_rec, fc):
    """Q19 on the validation questions: the Q02 terms, E_HS (Q02H), E_P1E (Q09's q09_core), E_SET, the frozen theta;
    the direction check of E_SET. Pure (no file). -> dict."""
    vk, S, n, ok, m, E04, P_q02 = q["vk"], q["S"], q["n"], q["ok"], q["m"], q["E04"], q["P"]
    Es = [E04[t] for t in r9.Q02_TERMS] + [core["E_hs"], qc9["E"], set_rec["E"]]
    P = R.fuse_probs(fc["params"]["theta"], m, Es, n, ok)
    has = set_rec["has"] & ok
    rho, nq = P7.hs_rho(set_rec["E"], P_q02, has)
    direction = P7.direction_record(rho, nq, fc["direction_ref"]["rho_eval"], RHO_FLOOR_VAL)
    return {"P": P, "has": has, "direction": direction, "csv_bytes": P7.probs_csv_bytes(vk["groups"], P, ok, n)}


def guard_outputs(a):
    existing = [n_ for n_ in APPLY_OUT if os.path.exists(os.path.join(OUT, n_))]
    if not existing:
        return None
    if not (a.rerun and a.reason):
        raise P19Error(f"apply outputs exist ({existing}); never overwritten without --rerun --reason TEXT")
    rp = os.path.join(OUT, "apply_report.json")
    if os.path.exists(rp) and read_json(rp).get("finalized"):
        raise P19Error("apply_report.json is finalized: the validation check is decided (a change needs a revision "
                       "and a new registration step)")
    return {"reason": a.reason, "replaced": {n_: pin_sha(os.path.join(OUT, n_)) for n_ in existing}}


def cmd_apply(a):
    t0 = time.time()
    if a.finalize:
        return apply_finalize()
    r9.blas_env_check(False)
    venv = venv_record()
    fc, vr, pre = apply_prerequisites()
    want_v = (fc.get("environment") or {}).get("versions") or {}
    if {k: venv[k] for k in ("python", "numpy", "scipy")} != {k: want_v.get(k) for k in ("python", "numpy", "scipy")}:
        raise P19Error(f".venv {venv} differs from frozen_config.json {want_v}")
    rerun = guard_outputs(a)
    # ---- 1 Q02 (prereg04's readers), 2 the question file (data/local: ids / counts / colours; before any guard)
    q = P7.q02_validation()
    q_sha = P7.check_question_file(q["vk"])
    vk, ok = q["vk"], q["ok"]
    # ---- 3 from here on no data/local file is opened (pilot_fuse imports hs_lib and its guard)
    PF = R.import_pilot_fuse()
    with np.load(R.P07_VAL_HS, allow_pickle=False) as z:
        vh = {k: z[k] for k in z.files}
    core = P7.apply_core(PF, q, vh, P7.load_hs_map(R.P07_HS_MAP), R.read_json(R.P07_FROZEN))
    with open(R.P07_PROBS, "rb") as f:
        ref07 = f.read().replace(b"\r\n", b"\n")
    _L1, F = R.import_p1()
    vp = R.read_val_chunks(vr, vk)
    ex32, chunk_shas = r9.read_val_ex(vr, vk)
    with np.load(R.P7.VAL_O4, allow_pickle=False) as z:
        o4z = {k: z[k] for k in z.files}
    fc08 = R.read_json(r9.P08_FROZEN)
    q8 = R.q08_core(F, q, core, vp, vh, o4z, R.load_p1_map(r9.P08_MAP), fc08)
    with open(r9.P08_PROBS, "rb") as f:
        ref08 = f.read().replace(b"\r\n", b"\n")
    fc9 = read_json(path_of("results/prereg09/rank/frozen_config.json"))
    mp = FW.load_map(path_of("results/prereg09/rank/p1e_map.npz"))
    qc9 = r9.q09_core(q, core, vp, ex32, fc9, mp)
    with open(os.path.join(P09_RANK, "rank_val_probs.csv"), "rb") as f:
        ref09 = f.read().replace(b"\r\n", b"\n")
    repro = {"Q02": bool(q["seam"]["passed"]), "Q02H": bool(core["csv_bytes"] == ref07),
             "Q08": bool(q8["csv_bytes"] == ref08), "Q09": bool(qc9["csv_bytes"] == ref09)}
    if not all(repro.values()):
        raise Stop(f"the earlier combinations do not reproduce byte for byte: {repro}; nothing written")
    v9 = dict(np.load(os.path.join(P09_RANK, "val_lp_p1e.npz"), allow_pickle=False))
    same = {"usable_equal_val_lp_p1e": bool(np.array_equal(np.asarray(v9["usable"], dtype=bool), qc9["usable"])),
            "p1_ok_equal_val_lp_p1e": bool(np.array_equal(np.asarray(v9["p1_ok"], dtype=bool), qc9["p1_ok"]))}
    same["passed"] = all(same.values())
    # ---- 4 the SET term (frozen S19, numpy float64) and Q19 with the frozen theta
    par_list = SF.load_models(OUT)
    E_set, lp_set, has_set = SF.question_set_term(par_list, vp["means"], vp["counts"], qc9["usable"], q["S"],
                                                  mp["mu_x"], mp["sd_x"])
    set_rec = {"E": E_set, "lp": lp_set, "has": has_set}
    qc = q19_core(q, core, qc9, set_rec, fc)
    fail_count = int(P7.hs_failures(vp["failed"], vp["n_moves"]).sum())
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, "rank_val_probs.csv")
    with open(out_csv + ".tmp", "wb") as f:
        f.write(qc["csv_bytes"])
    os.replace(out_csv + ".tmp", out_csv)
    vs_path = os.path.join(OUT, "val_set.npz")
    np.savez(vs_path + ".tmp.npz", format=np.array(FORMATS["val_set"]), combo=np.array("Q19"),
             question_id=np.array([qq for qq, _ in vk["groups"]]), lp=lp_set, has=has_set,
             usable=np.asarray(qc9["usable"], dtype=bool), scale=np.array(SCALE),
             models_sha256=np.array([SF.MODEL_SHA256[s] for s in SF.SEEDS]),
             mu_sd_from=np.array(P09_PINS["results/prereg09/rank/p1e_map.npz"]), chunk_sha256=np.array(chunk_shas))
    os.replace(vs_path + ".tmp.npz", vs_path)
    py = sys.executable
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r1 = subprocess.run([py, os.path.join(HERE, "decide.py"), out_csv, OUT], capture_output=True, text=True,
                        cwd=REPO, env=env)
    if r1.returncode != 0:
        raise P19Error(f"decide.py failed: {r1.stderr[-2000:]}")
    txt, formats_ok = [], True
    for pred in ("pred_A.csv", "pred_B.csv"):
        r2 = subprocess.run([py, os.path.join(HERE, "check_submission.py"), "rank", os.path.join(OUT, pred),
                             "--questions", P7.QUESTIONS_REL], capture_output=True, text=True, cwd=REPO, env=env)
        txt.append(f"$ check_submission.py rank results/prereg19/rank/{pred} --questions data/local/"
                   f"rank_val_questions.csv\n(exit {r2.returncode})\n{r2.stdout}{r2.stderr}")
        formats_ok = formats_ok and r2.returncode == 0
    with open(os.path.join(OUT, "check_submission.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(txt))
    audit, audit_rec = audit_state()
    usable, reasons = apply_decision(same, fail_count, repro, qc["direction"], formats_ok, audit)
    pred_9 = rf.rule_b_np(np.where(ok[:, None], qc9["P"], 0.0))
    pred_19 = rf.rule_b_np(np.where(ok[:, None], qc["P"], 0.0))
    dl = (pred_19 - pred_9)[ok]
    n_q = np.asarray(q["n"])
    by_n = {}
    for lo, hi in rf.N_BINS:
        sel = ok & (n_q >= lo) & (n_q <= hi)
        by_n[f"{lo}-{hi}"] = {"questions": int(sel.sum()), "rule_B_differs_from_Q09": int(((pred_19 != pred_9)
                                                                                         & sel).sum())}
    has2 = qc["has"] & qc9["has"]
    rho_e, n_e = P7.hs_rho(set_rec["E"], rf.softmax(np.asarray(qc9["E"], dtype=np.float64)), has2)
    rep = {"format": FORMATS["apply"], "what": "PREREG_19 2.2 apply: Q19 on the validation questions (no score)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), "mode": "real", "rerun": rerun,
           "combo": "Q19", "prerequisites": pre, "venv": venv, "reproduction": repro,
           "q02_reproduction": q["seam"], "same_as_09": same,
           "p1_failures": {"failed_or_no_move": fail_count, "failed": int(np.asarray(vp["failed"]).sum()),
                           "unusable_games": int((~qc9["usable"]).sum())},
           "direction": qc["direction"], "usable": usable, "usable_blocked_reasons": reasons, "audit": audit_rec,
           "set_term": {"questions": len(vk["groups"]), "with_usable_game": int(has_set.sum())},
           "record_only": {
               "rule_B_differs_from_Q09": int(((pred_19 != pred_9) & ok).sum()),
               "rule_B_differs_from_Q09_by_game_count": by_n,
               "level_difference_Q19_minus_Q09": {str(int(k)): int(v) for k, v in
                                                  zip(*np.unique(dl, return_counts=True))},
               "rho_E_SET_vs_E_P1E_expected_level": {"rho": rho_e, "questions": n_e},
               "Q09_direction": qc9["direction"]},
           "questions": len(vk["groups"]), "questions_without_parsed_game": int((~ok).sum()),
           "questions_without_usable_game": int((ok & ~qc["has"]).sum()),
           "pred_distribution": dict(zip(LEVELS, np.bincount(pred_19[ok], minlength=L).tolist())),
           "inputs_sha256": {"frozen_config.json": sha_lf(os.path.join(OUT, "frozen_config.json")),
                             **{SF.model_name(s): SF.sha256_file(p) for s, p in zip(SF.SEEDS, model_paths())},
                             "val_report.json (08)": sha_lf(r9.P08_VAL_REPORT),
                             "p1e_map.npz (09)": sha_raw(path_of("results/prereg09/rank/p1e_map.npz")),
                             "frozen_config.json (09)": sha_lf(path_of("results/prereg09/rank/frozen_config.json")),
                             "val_hs.npz": sha_raw(R.P07_VAL_HS), "questions": q_sha, "val_keys.npz": vk["sha256"],
                             "val_o4.npz": sha_raw(P7.VAL_O4), **q["vsha"]},
           "val_chunk_sha256": chunk_shas,
           "outputs_sha256": {n_: pin_sha(os.path.join(OUT, n_)) for n_ in APPLY_OUT[:6]},
           "decide_stdout": r1.stdout[-2000:], "tool": tool_state(), "environment": r9.environment_cpu(),
           "next": "commit; apply_audit.json by another agent (no labels, no scores) -> commit; apply --finalize"}
    write_json(os.path.join(OUT, "apply_report.json"), rep)
    log(f"apply Q19: reproduction {repro}; same as 09 {same['passed']}; rho {qc['direction']['rho']:.4f} (threshold "
        f"{qc['direction']['threshold']:.4f}); failed {fail_count}; formats {formats_ok}; rule B differs from Q09 on "
        f"{rep['record_only']['rule_B_differs_from_Q09']} questions")
    log(f"usable: {usable} {reasons}")
    return 0


# ---- the audit and finalize (as PREREG_09 2.7)
def audit_binding(audit, rep, audit_commit, apply_commit):
    probs = []
    if audit.get("format") != FORMATS["audit"]:
        probs.append(f"audit format {audit.get('format')!r} != {FORMATS['audit']!r}")
    if not isinstance(audit.get("program_errors_affecting_predictions"), list):
        probs.append("audit field program_errors_affecting_predictions is not a list")
    for n_ in AUDIT_BOUND_OUTPUTS:
        if (audit.get("outputs_sha256") or {}).get(n_) != (rep.get("outputs_sha256") or {}).get(n_):
            probs.append(f"audit outputs_sha256[{n_}] differs from apply_report.json")
    for n_ in AUDIT_BOUND_INPUTS:
        if (audit.get("inputs_sha256") or {}).get(n_) != (rep.get("inputs_sha256") or {}).get(n_):
            probs.append(f"audit inputs_sha256[{n_}] differs from apply_report.json")
    if (not apply_commit or not audit_commit
            or datetime.datetime.fromisoformat(audit_commit) <= datetime.datetime.fromisoformat(apply_commit)):
        probs.append(f"audit commit {audit_commit} is not later than apply_report.json's {apply_commit}")
    return probs


def audit_state(rep=None):
    p = os.path.join(OUT, "apply_audit.json")
    if not os.path.exists(p):
        return "missing", None
    if tracked_clean([p]):
        return "missing", {"note": "apply_audit.json exists but is not committed"}
    a = read_json(p)
    if a.get("format") != FORMATS["audit"]:
        return "missing", {"note": "apply_audit.json has the wrong format"}
    rec = {"sha256": sha_lf(p), "commit": commit_time(p)}
    if rep is not None:
        probs = audit_binding(a, rep, rec["commit"], commit_time(os.path.join(OUT, "apply_report.json")))
        if probs:
            rec["binding_problems"] = probs
            return "unbound", rec
    return ("bug" if a.get("program_errors_affecting_predictions") else "ok"), rec


def apply_finalize():
    rp = os.path.join(OUT, "apply_report.json")
    r9.require_committed([rp] + [os.path.join(OUT, f) for f in APPLY_OUT[:6]], "apply outputs (commit them first)")
    rep = read_json(rp)
    for n_, sha in rep["outputs_sha256"].items():
        if pin_sha(os.path.join(OUT, n_)) != sha:
            raise P19Error(f"{n_} changed since apply")
    if rep.get("finalized"):
        raise P19Error("apply_report.json is already finalized")
    audit, audit_rec = audit_state(rep)
    if audit == "missing":
        raise P19Error("apply_audit.json is not committed (finalize only after the audit)")
    if audit == "unbound":
        raise P19Error(f"apply_audit.json is not an audit of these outputs: {(audit_rec or {}).get('binding_problems')}")
    blocked = [x for x in rep.get("usable_blocked_reasons") or [] if not x.startswith("apply_audit.json")]
    if rep.get("usable") is False and blocked:
        usable, reasons = False, blocked
    elif audit == "bug":
        usable, reasons = False, ["apply_audit.json found a program error affecting the predictions"]
    else:
        usable, reasons = True, []
    rep["usable"], rep["usable_blocked_reasons"], rep["audit"] = usable, reasons, audit_rec
    rep["finalized"] = now_iso()
    rep["finalized_by_tool"] = tool_state()["files_sha256"]
    write_json(rp, rep)
    log(f"usable: {rep['usable']} {reasons}")
    return 0


# ================================================================ command line
def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason")
    p.add_argument("--revision", type=int)
    p = sub.add_parser("apply")
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason")
    a = ap.parse_args(argv)
    try:
        return {"freeze": cmd_freeze, "apply": cmd_apply}[a.cmd](a)
    except Stop as e:
        print(f"STOP (exit 1): {e}", file=sys.stderr)
        return 1
    except ERRORS as e:
        print(f"REFUSED (exit 2): {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
