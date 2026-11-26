"""prereg68 explore (PLAN.md, NOTES.md): competition 1 new questions, v2-9 (prereg67's 7-member score-level CNN
ensemble) with two more members X2, X3 (prereg62 X recipe, seeds 1 / 2, 400,000 steps) at weight 2 each.
CPU only, one process at a time. prereg67's code is imported and never edited.

    PY results/prereg68/explore/p68_eval.py seam --part pseudo|e      # NOTES.md 5 -> seam.json (commit before anything)
    PY results/prereg68/explore/p68_eval.py check-member --name X2|X3  # NOTES.md 3, 9 -> members.json
    PY results/prereg68/explore/p68_eval.py version                   # prints the version judge would use
    PY results/prereg68/explore/p68_eval.py e-use                     # committed E judgements only (NOTES.md 10)
    PY results/prereg68/explore/p68_eval.py judge --e-use N           # once -> judge_started.json, eval.json
    PY results/prereg68/explore/p68_eval.py apply --src data/test_v2_download/x/tests --pool public|private
    PY results/prereg68/explore/p68_eval.py assemble --src data/test_v2_download/x/tests

ens = prereg67's formula; E_ens = sum_m (w_m / W) zscore_rows(e_term(m, cfg_ens)) over
c49 1, S 1, B 1, p61 2, S1 2, S2 2, X 2, X2 2, X3 2 (members with weight 0 skipped, never loaded).
Nothing here opens the validation-answer folder or runs tools/score.py; no single-model E score is computed.
Audit data/advice/codex_1122b.md items 1-4: NOTES.md 9-12.
"""
import argparse
import datetime
import gc
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P67_DIR = os.path.join(REPO, "results", "prereg67", "explore")
if P67_DIR not in sys.path:
    sys.path.insert(0, P67_DIR)
import p67_eval as P67  # noqa: E402   (v2-9's code; imported, never edited)

if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p68_manifest as MF  # noqa: E402

P66, P63, P49, E43, E45, P39E = P67.P66, P67.P63, P67.P49, P67.E43, P67.E45, P67.P39E
BK, EV, st, L, Q42 = P67.BK, P67.EV, P67.st, P67.L, P67.Q42
_rel, read_json = P67._rel, P67.read_json

PLAN = os.path.join(HERE, "PLAN.md")
NOTES = os.path.join(HERE, "NOTES.md")
SEAM_JSON = os.path.join(HERE, "seam.json")
MEMBERS_JSON = os.path.join(HERE, "members.json")
EVAL_JSON = os.path.join(HERE, "eval.json")
JUDGE_STARTED = os.path.join(HERE, "judge_started.json")
JUDGE_FAILED = os.path.join(HERE, "judge_failed.json")
CODE = [os.path.join(HERE, f) for f in ("p68_eval.py", "p68_manifest.py", "p68_embed.py")]
OUT_APPLY = os.path.join(REPO, "results", "prereg68", "apply")
OFFICIAL_SRC = os.path.join(REPO, "data", "test_v2_download", "x", "tests")
V29_EVAL = os.path.join(P67_DIR, "eval.json")
V29_SEAM = os.path.join(P67_DIR, "seam.json")
V29_APPLY = os.path.join(REPO, "results", "prereg67", "apply")
X_RECORD = os.path.join(REPO, "results", "prereg62", "explore", "train_record_X.json")

BASE = tuple(P67.ORDER)                                   # v2-9: c49, S, B, p61, S1, S2, X
NEW = ("X2", "X3")
ORDER = BASE + NEW
WEIGHTS = dict(P67.WEIGHTS, X2=2.0, X3=2.0)
NEW_STEP = 400000
ARM_SEED = {"X2": 1, "X3": 2}
NEW_DIRS = {m: "data/prereg62/%s/emb/p60_step_400000_raw_tta1" % m for m in NEW}
POOLS = ("public", "private")
SIDECARS = tuple(os.path.splitext(f)[0] + ".provenance.json" for f in P67.FILES)
WE_ENS = P67.WE_ENS                                       # 14.0
CHECK_MIN = P67.CHECK_MIN                                 # 0.6406
TPE = datetime.timezone(datetime.timedelta(hours=8), "Asia/Taipei")   # UTC+8 all year (no DST in Taiwan)
CUTOFF = datetime.datetime(2026, 11, 24, 12, 0, 0, tzinfo=TPE)       # member files' deadline (NOTES.md 4, 11)
FILES = P67.FILES
V29_E_MEAN = 0.7379199600122529
ADOPT_MEAN = 0.005
NEED_POOLS = 16
BOOT, BOOT_SEED = 10000, 43
RAM_MIN_MB = 7000
EVAL_FORMAT = "c10-prereg68 eval v1"
START_FORMAT = "c10-prereg68 judge started v1"
APPLY_FORMAT = "c10-prereg68 apply v2"
GIT_ID = ["-c", "user.name=kuotunyu", "-c", "user.email=61350295+kuotunyu@users.noreply.github.com"]


def now_tpe():
    return datetime.datetime.now(TPE)


def mtime_tpe(p):
    return datetime.datetime.fromtimestamp(os.path.getmtime(p), TPE)


def write_json_atomic(path, obj):
    MF.write_json_atomic(path, obj)


# ================================================================ pure parts (tested in test_p68.py)
def weights_for(members):
    """Full 9-long weight list in ORDER; members not in the version get weight 0 (skipped)."""
    s = set(members)
    bad = s - set(ORDER)
    if bad:
        raise SystemExit("unknown members %s" % sorted(bad))
    return [WEIGHTS[m] if m in s else 0.0 for m in ORDER]


def member_state(e, disk, past_cutoff):
    """NOTES.md 11. e: members.json entry or None; disk: {present, before_cutoff, shas} of the member's 3 npz + 3
    sidecars now. Returns (state, problem): state in "in", "out-below-gate", "out-timeout", "wait", "problem"."""
    s = (e or {}).get("status")
    if s == "scored":
        if not e.get("pass"):
            return "out-below-gate", None
        if not disk.get("present") or disk.get("shas") != e.get("files_sha256"):
            return "problem", "passed member's files are missing or changed since check-member: restore them " \
                              "(no automatic downgrade to 8 members)"
        if not e.get("before_cutoff"):
            return "problem", "passed entry whose files were not made before the cutoff (inconsistent record)"
        return "in", None
    if s not in (None, "invalid"):
        return "problem", "unknown members.json status %r" % s
    if s == "invalid":
        why = "checked invalid at %s (%s)" % (e.get("checked_at"), "; ".join(e.get("problems") or [])[:300])
        if disk.get("present") and disk.get("before_cutoff") and disk.get("shas") != e.get("files_sha256"):
            return "problem", "%s; files changed since: run check-member again" % why
        if past_cutoff:                                     # known invalid stays a named problem (audit 1122d 3)
            return "problem", ("%s; not repaired and re-checked by the cutoff: judge refuses (main session "
                               "decides)" % why)
        return "wait", None                                # may be repaired before the cutoff
    if disk.get("present"):
        if not disk.get("before_cutoff"):
            return "out-timeout", None                     # finished after the cutoff
        return "problem", "vectors present before the cutoff but not checked: run check-member"
    return ("out-timeout", None) if past_cutoff else ("wait", None)


def pick_version(entries, disk, past_cutoff):
    """Returns (members, problems, wait, states). problems: stop; wait: the judge refuses until the member is
    checked / dropped; members: v2-9 + the new members whose state is 'in'."""
    problems, wait, states, use = [], [], {}, list(BASE)
    for nm in NEW:
        s, why = member_state(entries.get(nm), disk.get(nm, {}), past_cutoff)
        states[nm] = s
        if s == "problem":
            problems.append("%s: %s" % (nm, why))
        elif s == "wait":
            wait.append(nm)
        elif s == "in":
            use.append(nm)
    return [m for m in ORDER if m in use], problems, wait, states


def version_name(members):
    s = set(members)
    if s == set(ORDER):
        return "9"
    if s == set(BASE) | {"X2"}:
        return "8-X2"
    if s == set(BASE) | {"X3"}:
        return "8-X3"
    if s == set(BASE):
        return "7 (v2-9; nothing to judge)"
    raise SystemExit("unexpected member set %s" % sorted(s))


def judge_rule(new, old):
    d = np.asarray(new, np.float64) - np.asarray(old, np.float64)
    lo, hi = Q42.bootstrap(d, BOOT, BOOT_SEED)
    r = {"mean": float(d.mean()), "ci95": [lo, hi], "pools_higher": int((d > 0).sum())}
    r["adopt"] = bool(r["mean"] >= ADOPT_MEAN and lo > 0 and r["pools_higher"] >= NEED_POOLS)
    return r


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo] + list(args), capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def e_uses_so_far(repo=REPO):
    """NOTES.md 10: e_pools_use_number of results/prereg*/explore/eval*.json as committed in HEAD (git show; the
    working tree is not read), prereg68's own eval.json excluded."""
    own = "results/prereg68/explore/eval.json"
    r = _git(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", "results")
    if r.returncode != 0:
        raise SystemExit("git ls-tree failed: %s" % r.stderr.strip())
    nums = {}
    for f in sorted(r.stdout.split("\n")):
        p = f.strip().split("/")
        if len(p) != 4 or not p[1].startswith("prereg") or p[2] != "explore" or not \
                (p[3].startswith("eval") and p[3].endswith(".json")) or f.strip() == own:
            continue
        c = _git(repo, "show", "HEAD:" + f.strip())
        if c.returncode != 0:
            raise SystemExit("git show HEAD:%s failed: %s" % (f.strip(), c.stderr.strip()))
        try:
            d = json.loads(c.stdout)
        except ValueError:
            continue
        if isinstance(d, dict) and "e_pools_use_number" in d:
            nums[f.strip()] = int(d["e_pools_use_number"])
    mx = max(nums.values()) if nums else 0
    vals = sorted(nums.values())
    return {"files": nums, "max": mx, "count": mx, "source": "git HEAD",
            "consistent": len(set(vals)) == len(vals) and vals == list(range(4, mx + 1))}


def ever_committed(path, repo=REPO):
    """True if any commit reachable from HEAD touched the path (a committed start record / eval that was deleted
    later still counts)."""
    rel = os.path.relpath(os.path.abspath(path), repo).replace("\\", "/")
    r = _git(repo, "log", "--format=%H", "-1", "--", rel)
    if r.returncode != 0:
        raise SystemExit("git log failed for %s (cannot tell whether it was committed): %s" % (rel, r.stderr.strip()))
    return bool(r.stdout.strip())


def judge_guard():
    """NOTES.md 10: stop if the judgement has started / finished before (working tree or git history)."""
    for f in (JUDGE_STARTED, EVAL_JSON):
        if os.path.exists(f) or ever_committed(f):
            raise SystemExit("%s exists (now or in git history): the judgement is made once; do not rerun"
                             % _rel(f))


def claim_start(rec):
    """Exclusive, atomic creation of judge_started.json (O_CREAT | O_EXCL): a second process loses here."""
    try:
        fd = os.open(JUDGE_STARTED, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("judge_started.json was created by another run: stop (the judgement is made once)")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def commit_start(e_use):
    """Commit the start record at once (only that path). Returns True only if the commit succeeded and the file is
    now tracked and identical to HEAD (audit 1122d item 1: the caller stops before any new score otherwise)."""
    rel = _rel(JUDGE_STARTED)
    a = _git(REPO, "add", "--", rel)
    c = _git(REPO, *GIT_ID, "commit", "-m", "prereg68 judge started (E use %d): start record" % e_use, "--", rel)
    ok = a.returncode == 0 and c.returncode == 0 and L.git_committed(JUDGE_STARTED) and ever_committed(JUDGE_STARTED)
    print(json.dumps({"start_record_committed": ok, "git": (c.stdout + c.stderr).strip()[-300:]}), flush=True)
    return ok


# ================================================================ member files
def v29_eval():
    ev = read_json(V29_EVAL)
    if ev["members"] != list(BASE) or ev["means"]["ens"] != V29_E_MEAN or not ev["adopt"]:
        raise SystemExit("results/prereg67/explore/eval.json is not the adopted 7-member v2-9")
    for m in BASE:
        if ev["weights"][m] != WEIGHTS[m]:
            raise SystemExit("v2-9 weight of %s differs" % m)
    if ev["w_E"] != WE_ENS:
        raise SystemExit("v2-9 w_E differs")
    return ev


def entries68():
    return read_json(MEMBERS_JSON) if os.path.exists(MEMBERS_JSON) else {}


def member_dir(m, entries=None):
    if m in NEW:
        return os.path.join(REPO, NEW_DIRS[m])        # fixed (no --dir): NOTES.md 9
    return P67.member_dir(m)          # frozen.json dirs (p61 / X: the same dirs as prereg67 members.json)


def member_shas(m, entries, ev29):
    if m in NEW:
        return {f: entries[m]["files_sha256"][f] for f in FILES}
    return ev29["member_files"][m]


def all_files(d):
    return [os.path.join(d, f) for f in FILES + SIDECARS]


def disk_state(d):
    paths = all_files(d)
    present = all(os.path.exists(p) for p in paths)
    return {"present": present,
            "before_cutoff": present and all(mtime_tpe(p) <= CUTOFF for p in paths),
            "shas": {os.path.basename(p): L.sha256_file(p) for p in paths} if present else None}


def arm_record_path(arm):
    return os.path.join(REPO, "data", "prereg62", arm, "train_record_%s.json" % arm)


def arm_ckpt_path(arm):
    return os.path.join(REPO, "data", "prereg62", arm, "ckpt", "step_%06d.pt" % NEW_STEP)


def identity_problems(arm, rec, xrec, ckpt_sha):
    """Pure: the arm's training record vs prereg62 X's (only the seed may differ) and the checkpoint file's sha."""
    p = []
    if not rec.get("finished") or rec.get("steps") != NEW_STEP or rec.get("smoke"):
        p.append("record not finished / steps %r / smoke %r" % (rec.get("steps"), rec.get("smoke")))
    c, cx = rec.get("config", {}), xrec["config"]
    if c.get("variant") != cx["variant"] or c.get("spec") != cx["spec"]:
        p.append("variant / spec differ from X's")
    a, b = dict(c.get("p60", {})), dict(cx["p60"])
    if a.get("seed") != ARM_SEED[arm]:
        p.append("seed %r != %d" % (a.get("seed"), ARM_SEED[arm]))
    a.pop("seed", None)
    b.pop("seed", None)
    if a != b:
        p.append("p60 config differs from X's beyond the seed: %s" % sorted(k for k in set(a) | set(b)
                                                                          if a.get(k) != b.get(k)))
    want = rec.get("checkpoints", {}).get("step_%06d.pt" % NEW_STEP)
    if ckpt_sha is None:
        p.append("checkpoint file missing")
    elif want != ckpt_sha:
        p.append("checkpoint file sha256 %s != the record's %s" % (ckpt_sha[:12], str(want)[:12]))
    return p


def arm_identity(arm):
    """NOTES.md 9 (a). Returns (ident, problems)."""
    rp, cp = arm_record_path(arm), arm_ckpt_path(arm)
    if not os.path.exists(rp):
        return None, ["%s missing" % _rel(rp)]
    rec, xrec = read_json(rp), read_json(X_RECORD)
    sha = L.sha256_file(cp) if os.path.exists(cp) else None
    prob = identity_problems(arm, rec, xrec, sha)
    return {"record": _rel(rp), "record_sha256": L.sha256_file(rp), "ckpt": _rel(cp), "ckpt_sha256": sha,
            "seed": rec.get("config", {}).get("p60", {}).get("seed"), "steps": rec.get("steps"),
            "players_T_sha256": rec.get("config", {}).get("p60", {}).get("T_sha256_sorted_ids")}, prob


def other_ckpt_shas(arm):
    """Checkpoint sha256 of every other member (v2-9's from their store_E.npz, X's from its record, the other new
    arm's from its record if present)."""
    out = {}
    for m in BASE:
        with np.load(os.path.join(member_dir(m), "store_E.npz"), allow_pickle=False) as z:
            out[m] = str(z["ckpt_sha256"])
    out["X(record)"] = read_json(X_RECORD)["checkpoints"]["step_%06d.pt" % NEW_STEP]
    for o in NEW:
        if o != arm and os.path.exists(arm_record_path(o)):
            out[o] = read_json(arm_record_path(o)).get("checkpoints", {}).get("step_%06d.pt" % NEW_STEP)
    return out


def distinct_problems(ckpt_sha, others):
    return ["checkpoint sha256 equals %s's" % k for k, v in sorted(others.items()) if v == ckpt_sha]


def array_problems(f, emb, count):
    p = []
    if np.asarray(count).ndim != 1 or len(count) != len(emb):
        p.append("%s count shape %s vs %d vectors" % (f, np.shape(count), len(emb)))
    elif not (np.asarray(count) > 0).all():
        p.append("%s has games without a sample" % f)
    if not np.isfinite(emb).all():
        p.append("%s has non-finite values" % f)
    return p


def validate_new(name, src=OFFICIAL_SRC):
    """NOTES.md 9: everything check-member verifies before std10. Returns (record, problems)."""
    d = member_dir(name)
    files = P67.check_files(name, d, step=NEW_STEP)
    prob = list(files["problems"])
    ident, ip = arm_identity(name)
    prob += ip
    ck = ident["ckpt_sha256"] if ident else None
    if ck:
        prob += distinct_problems(ck, other_ckpt_shas(name))
    c49e = os.path.join(member_dir("c49"), "store_E.npz")
    with np.load(c49e, allow_pickle=False) as r:
        c49_players = str(r["players_sha256_sorted_ids"])
    for f in FILES:
        with np.load(os.path.join(d, f), allow_pickle=False) as z:
            if str(z["ckpt_sha256"]) != ck:
                prob.append("%s checkpoint sha256 is not the arm's step_400000.pt" % f)
            if f == "store_E.npz" and str(z["players_sha256_sorted_ids"]) != c49_players:
                prob.append("store_E player-set hash differs from c49's")
            prob += array_problems(f, z["emb"], z["count"])
        gc.collect()
    if ck:
        prob += MF.verify_store(os.path.join(d, "store_E.npz"), ck)
        for pool in POOLS:
            prob += ["pool_%s: %s" % (pool, x) for x in MF.verify_pool(os.path.join(d, "pool_%s.npz" % pool),
                                                                        src, pool, ck)]
    return {"files": files, "identity": ident}, prob


def version_from_disk(now=None):
    entries = entries68()
    disk = {nm: disk_state(member_dir(nm)) for nm in NEW}
    now = now or now_tpe()
    members, problems, wait, states = pick_version(entries, disk, now > CUTOFF)
    return members, problems, wait, {
        "now": now.isoformat(timespec="seconds"), "cutoff": CUTOFF.isoformat(), "states": states,
        "disk": {k: {"present": v["present"], "before_cutoff": v["before_cutoff"]} for k, v in disk.items()},
        "entries": {k: {"status": v.get("status"), "pass": v.get("pass"), "std10": v.get("std10_mean")}
                    for k, v in entries.items()}}


# ================================================================ scores
def pseudo_scores(ctx, members, entries, cfg_x, ev29):
    """std10 pseudo pools, ensemble of `members` (weights_for; weight-0 members never loaded)."""
    w = weights_for(members)
    on = [(m, wi) for m, wi in zip(ORDER, w) if wi]
    Zs = {}
    for m, _wi in on:
        d = member_dir(m, entries)
        if P67.file_shas(d) != member_shas(m, entries, ev29):
            raise SystemExit("%s vectors differ from their record" % m)
        Zs[m] = P67.pseudo_Z(ctx, P67.pseudo_emb(m, d, ctx["cols"]), cfg_x)
        gc.collect()
    per = []
    for i, p in enumerate(ctx["pools"]):
        E_ens = P67.ensemble([Zs[m][i] for m, _ in on], [wi for _, wi in on])
        S = P67.score_mat(p["G"], E_ens, p["z1"], p["zm"], p["zs"], p["zr"], p["ks"], WE_ENS)
        per.append(float(st.score(S, p["owner"])))
    return per


def e_ensembles(ctx, member_sets, entries, cfg_x, ev29):
    """E_ens per pool for each member set (same arithmetic as prereg67 e_ensemble: (w/W) z, summed in ORDER,
    weight-0 members skipped). Each member's vectors are loaded once, sha256 checked."""
    ws = [weights_for(s) for s in member_sets]
    Ws = [float(sum(x for x in w if x)) for w in ws]
    accs = [[None] * len(ctx["pools"]) for _ in ws]
    okc = [np.ones(len(p["q_rows"]), bool) for p in ctx["pools"]]
    used = {}
    for j, m in enumerate(ORDER):
        if not any(w[j] for w in ws):
            continue
        path = os.path.join(member_dir(m, entries), "store_E.npz")
        sha = L.sha256_file(path)
        if sha != member_shas(m, entries, ev29)["store_E.npz"]:
            raise SystemExit("%s store_E differs from its record" % m)
        used[m] = {"file": _rel(path), "sha256": sha}
        e = E43.load_emb(path)
        for i, p in enumerate(ctx["pools"]):
            E, ok = BK.e_term(p, e, cfg_x)
            okc[i] &= np.asarray(ok, bool)
            Z = st.zscore_rows(E, ok)
            for w, W, acc in zip(ws, Ws, accs):
                if w[j]:
                    t = (float(w[j]) / W) * Z
                    acc[i] = t if acc[i] is None else acc[i] + t
            del E, ok, Z
        del e
        gc.collect()
    return accs, okc, used


def v29_E_check(per, ev29):
    want = ev29["per_pool"]["ens"]
    return bool(per == want and float(np.mean(per)) == V29_E_MEAN)


# ================================================================ commands
def cmd_seam(a):
    t0 = time.time()
    ev29 = v29_eval()
    seam = read_json(SEAM_JSON) if os.path.exists(SEAM_JSON) else {"format": "c10-prereg68 seam v1"}
    entries = {}                                        # X2 / X3 weight 0: never loaded
    ce = None
    if a.part == "pseudo":
        ctx = P67.pseudo_context()
        ce = P67.cfg_ens(ctx["cfg"])
        s67 = read_json(V29_SEAM)["pseudo"]["recipe_5m_tilt2_085cc_wE14"]
        per5 = pseudo_scores(ctx, list(P67.FIXED), entries, ce, ev29)
        per7 = pseudo_scores(ctx, list(BASE), entries, ce, ev29)
        res = {"five_fixed_members": {"per_pool": per5, "mean": float(np.mean(per5)),
                                      "want_per_pool": s67["float64_per_pool"], "want_mean": s67["float64_mean"],
                                      "identical": bool(per5 == s67["float64_per_pool"]
                                                        and float(np.mean(per5)) == s67["float64_mean"])},
               "seven_v29_members_recorded": {"per_pool": per7, "mean": float(np.mean(per7))},
               "weights_X2_X3": [0.0, 0.0]}
        res["ok"] = res["five_fixed_members"]["identical"]
        print(json.dumps({"five_identical": res["ok"], "five_mean": res["five_fixed_members"]["mean"],
                          "seven_mean": res["seven_v29_members_recorded"]["mean"]}), flush=True)
        key = "pseudo"
    else:
        ctx = P67.e_context()
        ce = P67.cfg_ens(ctx["cfg"])
        (E7,), okc, used = e_ensembles(ctx, [list(BASE)], entries, ce, ev29)
        not_ok = int(sum(int((~o).sum()) for o in okc))
        per, _byk = P67.e_scores(ctx, E7, WE_ENS)
        res = {"per_pool": per, "mean": float(np.mean(per)), "want": V29_E_MEAN,
               "identical": v29_E_check(per, ev29), "ok_ens_false_questions": not_ok, "vectors": used,
               "weights": weights_for(BASE), "order": list(ORDER)}
        res["ok"] = bool(res["identical"] and not not_ok)
        print(json.dumps({k: res[k] for k in ("mean", "identical", "ok_ens_false_questions")}), flush=True)
        key = "E"
    res.update(cfg_ens=ce, git_head=L.git_head(), peak_rss_mb=P63.peak_mb(), seconds=round(time.time() - t0, 1))
    seam = read_json(SEAM_JSON) if os.path.exists(SEAM_JSON) else seam
    seam[key] = res
    seam["ok"] = bool(seam.get("pseudo", {}).get("ok") and seam.get("E", {}).get("ok"))
    L.write_json(SEAM_JSON, seam)
    if not res["ok"]:
        raise SystemExit("seam %s failed: stop (do not judge)" % a.part)
    return 0


def cmd_check_member(a):
    """NOTES.md 3, 9, 11. Invalid files -> status 'invalid' (no std10; may be checked again after a repair).
    Valid files -> std10 -> status 'scored' (final; pass iff std10 >= 0.6406)."""
    t0 = time.time()
    seam = read_json(SEAM_JSON) if os.path.exists(SEAM_JSON) else {}
    if not seam.get("ok") or not L.git_committed(SEAM_JSON):
        raise SystemExit("seam.json is not ok / not committed")
    entries = entries68()
    old = entries.get(a.name)
    if old is not None and old.get("status") != "invalid":
        raise SystemExit("%s already scored (members.json): the check is made once" % a.name)
    d = member_dir(a.name)
    disk = disk_state(d)
    if not disk["present"]:
        raise SystemExit("%s: vectors / provenance sidecars missing in %s: not ready (nothing written)"
                         % (a.name, _rel(d)))
    if not disk["before_cutoff"]:
        raise SystemExit("%s: files made after the cutoff %s: dropped as late (nothing written)" % (a.name, CUTOFF))
    info, prob = validate_new(a.name)
    rec = {"name": a.name, "dir": _rel(d), "files": info["files"], "identity": info["identity"],
           "files_sha256": disk["shas"], "before_cutoff": True, "threshold": CHECK_MIN,
           "checked_at": now_tpe().isoformat(timespec="seconds"), "problems": prob}
    if old is not None:
        rec["earlier_invalid_attempts"] = old.get("earlier_invalid_attempts", []) + \
            [{k: old.get(k) for k in ("checked_at", "problems", "files_sha256")}]
    print(json.dumps({"name": a.name, "valid": not prob, "problems": prob}), flush=True)
    if prob:
        rec.update({"status": "invalid", "pass": None, "std10_mean": None})
    else:
        if disk_state(d)["shas"] != disk["shas"]:
            raise SystemExit("files changed while being checked: stop")
        ctx = P67.pseudo_context()
        per = P67.pseudo_single(ctx, d, a.name)
        rec["std10_per_pool"] = per
        rec["std10_mean"] = float(np.mean(per))
        rec["status"] = "scored"
        rec["pass"] = bool(rec["std10_mean"] >= CHECK_MIN)
    rec.update(git_head=L.git_head(), peak_rss_mb=P63.peak_mb(), seconds=round(time.time() - t0, 1))
    entries = entries68()
    entries[a.name] = rec
    write_json_atomic(MEMBERS_JSON, entries)
    print(json.dumps({k: rec.get(k) for k in ("name", "status", "std10_mean", "pass", "peak_rss_mb", "seconds")}))
    return 0


def cmd_version(a):
    members, problems, wait, info = version_from_disk()
    print(json.dumps({"members": members, "version": version_name(members), "problems": problems, "wait": wait,
                      "info": info}, indent=1))
    return 0


def judge_preflight(a):
    for f in [PLAN, NOTES, SEAM_JSON, MEMBERS_JSON] + CODE:
        if not os.path.exists(f) or not L.git_committed(f):
            raise SystemExit("%s is missing / not committed" % _rel(f))
    if not read_json(SEAM_JSON).get("ok"):
        raise SystemExit("seam not ok")
    ev29 = v29_eval()
    uses = e_uses_so_far()
    if not uses["consistent"]:
        raise SystemExit("E use counters disagree: %s" % uses)
    if a.e_use != uses["count"] + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before (committed)" % (a.e_use,
                                                                                                  uses["count"]))
    members, problems, wait, vinfo = version_from_disk()
    if problems:
        raise SystemExit("; ".join(problems))
    if wait:
        raise SystemExit("%s neither checked nor dropped and the cutoff %s has not passed: wait" % (wait, CUTOFF))
    if set(members) == set(BASE):
        raise SystemExit("no new member passed: nothing to judge, v2-9 stays")
    entries = entries68()
    for m in members:
        if m in NEW:
            ident, ip = arm_identity(m)
            if ip or ident["ckpt_sha256"] != entries[m]["identity"]["ckpt_sha256"]:
                raise SystemExit("%s identity changed since check-member: %s" % (m, ip))
    return ev29, uses, members, vinfo, entries


def cmd_judge(a):
    """NOTES.md 6, 10. Order: guard -> preflight -> v2-9 rebuilt (seam) -> start record (exclusive, committed)
    -> the one score of the new version -> eval.json (atomic)."""
    t0 = time.time()
    judge_guard()
    ev29, uses, members, vinfo, entries = judge_preflight(a)
    vname = version_name(members)
    print(json.dumps({"version": vname, "members": members}), flush=True)
    ctx = P67.e_context()
    ce = P67.cfg_ens(ctx["cfg"])
    (E29, Enew), okc, used = e_ensembles(ctx, [list(BASE), members], entries, ce, ev29)
    not_ok = int(sum(int((~o).sum()) for o in okc))
    if not_ok:
        raise SystemExit("%d E questions without a vector in some member: stop" % not_ok)
    base_per, base_byk = P67.e_scores(ctx, E29, WE_ENS)
    if not v29_E_check(base_per, ev29):
        raise SystemExit("seam: v2-9 rebuilt in the judgement differs from prereg67 eval.json")
    del E29
    gc.collect()
    mfiles = {m: member_shas(m, entries, ev29) for m in members}
    start = {"format": START_FORMAT, "e_pools_use_number": a.e_use, "version": vname, "members": members,
             "member_files": mfiles, "members_json_sha256": st.sha256_lf(MEMBERS_JSON), "git_head": L.git_head(),
             "pid": os.getpid(), "started_at": now_tpe().isoformat(timespec="seconds"),
             "note": "written after the v2-9 seam passed and before the one score of the new version; never "
                     "deleted; its presence (now or in git history) blocks any further judge run"}
    judge_guard()
    claim_start(start)
    if not commit_start(a.e_use):
        raise SystemExit("judge_started.json could not be committed: stop before the new version is scored (the "
                         "start record stays; the judgement is not made; report, do not rerun)")
    try:
        per, byk = P67.e_scores(ctx, Enew, WE_ENS)
        j = judge_rule(per, base_per)
        bk_e, bk_b = EV.mean_by_k(byk), EV.mean_by_k(base_byk)
        rec = {"format": EVAL_FORMAT, "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
               "earlier_e_judgements": uses, "version": vname, "members": members,
               "weights": {m: WEIGHTS[m] for m in members}, "w_E": WE_ENS, "cfg_ens": ce,
               "member_vectors": used, "member_files": mfiles,
               "version_info": vinfo, "ok_ens_false_questions": not_ok, "seam_v29_E_identical": True,
               "seam_json_sha256": st.sha256_lf(SEAM_JSON), "members_json_sha256": st.sha256_lf(MEMBERS_JSON),
               "judge_started_sha256": st.sha256_lf(JUDGE_STARTED),
               "git_head": L.git_head(), "means": {"ens": float(np.mean(per)), "v29": float(np.mean(base_per))},
               "per_pool": {"ens": per, "v29": base_per}, "by_k": {"ens": bk_e, "v29": bk_b},
               "by_k_diff": {k: bk_e[k] - bk_b[k] for k in bk_e},
               "ens_minus_v29": {x: j[x] for x in ("mean", "ci95", "pools_higher")},
               "rule": "adopt if mean(ens - v2-9) >= %.3f and ci95 low > 0 and higher in >= %d of %d pools (pool "
                       "bootstrap %d, seed %d)" % (ADOPT_MEAN, NEED_POOLS, len(per), BOOT, BOOT_SEED),
               "adopt": j["adopt"],
               "note": "single-model E scores are not computed (PLAN); prereg66 F1 holds for both systems",
               "peak_rss_mb": P63.peak_mb(), "seconds": round(time.time() - t0, 1)}
        if os.path.exists(EVAL_JSON):
            raise SystemExit("eval.json appeared during the run: stop")
        write_json_atomic(EVAL_JSON, rec)
    except BaseException as e:  # noqa: BLE001  (the start record stays; never judged again automatically)
        try:
            write_json_atomic(JUDGE_FAILED, {"error": repr(e)[:2000], "at": now_tpe().isoformat(timespec="seconds"),
                                             "note": "the judgement started and did not finish: report, do not "
                                                     "rerun"})
        except Exception:  # noqa: BLE001
            pass
        raise
    print(json.dumps({x: rec[x] for x in ("version", "means", "ens_minus_v29", "by_k_diff", "adopt")}, indent=1))
    return 0


def adopted_eval():
    """NOTES.md 12: only this round's committed, adopted eval.json (with its committed start record)."""
    for f in (EVAL_JSON, JUDGE_STARTED, MEMBERS_JSON):
        if not os.path.exists(f) or not L.git_committed(f):
            raise SystemExit("%s is missing / not committed" % _rel(f))
    ev, start = read_json(EVAL_JSON), read_json(JUDGE_STARTED)
    p = []
    if ev.get("format") != EVAL_FORMAT or start.get("format") != START_FORMAT:
        p.append("format")
    if not ev.get("adopt"):
        raise SystemExit("eval.json does not adopt the ensemble: keep v2-9")
    if ev.get("members") != start.get("members") or ev.get("version") != start.get("version") or \
            version_name(ev.get("members", [])) != ev.get("version"):
        p.append("members / version differ between eval.json and judge_started.json")
    if ev.get("e_pools_use_number") != start.get("e_pools_use_number"):
        p.append("E use number differs from the start record")
    if ev.get("weights") != {m: WEIGHTS[m] for m in ev.get("members", [])} or ev.get("w_E") != WE_ENS:
        p.append("weights / w_E are not prereg68's")
    if ev.get("members_json_sha256") != st.sha256_lf(MEMBERS_JSON):
        p.append("members.json changed since the judgement")
    if ev.get("judge_started_sha256") != st.sha256_lf(JUDGE_STARTED):
        p.append("judge_started.json changed since the judgement")
    if p:
        raise SystemExit("eval.json not usable: %s" % p)
    return ev, st.sha256_lf(EVAL_JSON)


def pool_terms(src, pool):
    """A66's non-CNN terms on an official pool, exactly as prereg67 apply (p67_eval.py lines 584-628)."""
    base = E45.v26()
    rec, items, cand_ids, qids = E43.official_rec(src, pool)
    name = {"public": "pub2", "private": "priv2"}[pool]
    colour = np.array([c for _s, c in items])
    ks = np.array([len(g) for g in rec["q_rows"]], np.int64)
    if not set(ks.tolist()) <= {1, 2, 3, 5}:
        raise SystemExit("question game counts outside {1, 2, 3, 5}")
    Pset, Lset = P39E.load_set(name), P39E.load_set("L")
    if len(Pset["games"]) != len(items) or any(g["color"] != c for g, (_s, c) in zip(Pset["games"], items)):
        raise SystemExit("the P1 list of the pool does not match the pool's games")
    models = P39E.fit_models(Lset)
    del Lset
    if P39E.models_sha(models) != base["lda_sha256"]:
        raise SystemExit("the L LDA refit differs from prereg43 tune.json")
    z1, has = P39E.p1_terms(Pset["X"], Pset["usable"], colour, rec["cand_rows"], rec["q_rows"], models,
                            E43.P1_PW)[E43.P1_VARIANT]
    del models
    games, means, counts, usable = P63.load_p1_full(name)
    if len(games) != len(items) or any(g["color"] != c for g, (_s, c) in zip(games, items)):
        raise SystemExit("the P1 list of the pool does not match the pool's games")
    if not (np.array_equal(usable, Pset["usable"]) and np.array_equal(means[:, 0, :], Pset["X"])):
        raise SystemExit("Carbon's P1 loader differs from the v2-7 P1 set on %s: stop" % name)
    del Pset
    seam66 = read_json(P67.V28_SEAM)
    mods = P63.load_multi(seam66["p1multi_sha256"])
    sp = P66.load_sp(seam66["sp128"]["sha256"])
    head = P66.load_head(P66.C49_HEAD)
    q09, s25 = P63.load_models()
    cg, qg = rec["cand_rows"], rec["q_rows"]
    zm = P63.z_multi_of(P63.project_multi(means, mods), usable, cg, qg)
    zsp = P66.pool_whiten_proto_z(P66.project_sp128(means, sp), usable, cg, qg, P66.R["c13_pool_lam"])
    path49 = P49.c49_file(pool)
    vec49 = P49.check_c49_file(path49, pool, base["tune"]["c_star"])
    e49 = E43.pool_emb(path49, items, vec49["ckpt_sha256"])

    def emb_fn(rows):
        return e49.emb[rows], np.asarray(e49.count)[rows]

    zr = P66.rank_z_pool(cg, qg, means, counts, usable, emb_fn, q09, s25, head)
    del e49
    w_G, _kG = EV.g_combo()
    G = st.fuse([rec["T"][n] for n in BK.ps.TERMS], list(w_G))
    return {"rec": rec, "items": items, "cand_ids": cand_ids, "qids": qids, "colour": colour, "ks": ks,
            "G": G, "z1": z1, "zm": zm, "zs": zsp, "zr": zr, "has": has, "usable": usable, "cfg": base["cfg"]}


def items_match_manifest(items, rows):
    """Pure: the apply's item list (sgf, colour) vs the bound manifest rows (sgf sha256, colour)."""
    return len(items) == len(rows) and all(MF.sha_text(s) == r[5] and c.upper() == r[4]
                                           for (s, c), r in zip(items, rows))


def cmd_apply(a):
    t0 = time.time()
    ev, ev_sha = adopted_eval()
    ev29 = v29_eval()
    entries = entries68()
    members = ev["members"]
    for m in members:
        if P67.file_shas(member_dir(m, entries)) != ev["member_files"][m]:
            raise SystemExit("%s vectors differ from eval.json" % m)
    rows = MF.rebuild(a.src, a.pool)
    prov = {}
    for m in members:
        if m in NEW:
            ck = entries[m]["identity"]["ckpt_sha256"]
            path = os.path.join(member_dir(m), "pool_%s.npz" % a.pool)
            pr = MF.verify_pool(path, a.src, a.pool, ck, rows)
            if pr:
                raise SystemExit("%s pool_%s provenance: %s" % (m, a.pool, pr))
            prov[m] = {"sidecar_sha256": L.sha256_file(MF.sidecar_path(path)), "manifest_sha256": MF.manifest_sha(rows)}
    t = pool_terms(a.src, a.pool)
    rec, items, cand_ids, qids, ks = t["rec"], t["items"], t["cand_ids"], t["qids"], t["ks"]
    if prov and not items_match_manifest(items, rows):
        raise SystemExit("the apply's game list differs from the X2 / X3 manifest: stop")
    ce = P67.cfg_ens(t["cfg"])
    Z, okc, used = {}, np.ones(len(rec["q_rows"]), bool), {}
    for m in ORDER:
        if m not in members and m not in BASE:
            continue
        path = os.path.join(member_dir(m, entries), "pool_%s.npz" % a.pool)
        e = EV.Emb.load(path, colour=t["colour"])
        if len(e.emb) != len(items):
            raise SystemExit("%s: %d vectors for %d games" % (m, len(e.emb), len(items)))
        if m in NEW and e.sha != entries[m]["identity"]["ckpt_sha256"]:
            raise SystemExit("%s pool file's checkpoint is not the arm's" % m)
        E, ok = BK.e_term(rec, e, ce)
        okc &= np.asarray(ok, bool)
        Z[m] = st.zscore_rows(E, ok)
        used[m] = {"file": _rel(path), "sha256": L.sha256_file(path), "ckpt_sha256": e.sha}
        del e, E
        gc.collect()
    if not okc.all():
        raise SystemExit("%d questions without a vector in some member: stop" % int((~okc).sum()))

    def scores(ms):
        w = weights_for(ms)
        on = [(m, wi) for m, wi in zip(ORDER, w) if wi]
        E_ens = P67.ensemble([Z[m] for m, _ in on], [wi for _, wi in on])
        return P67.score_mat(t["G"], E_ens, t["z1"], t["zm"], t["zs"], t["zr"], ks, WE_ENS)

    data29 = P39E.top5_bytes(qids, st.top5(scores(list(BASE)), cand_ids))
    with open(os.path.join(V29_APPLY, a.pool, "top5.csv"), "rb") as f:
        if P63.lf(f.read()) != P63.lf(data29):
            raise SystemExit("seam: v2-9 rebuilt on %s differs from results/prereg67/apply/%s/top5.csv"
                             % (a.pool, a.pool))
    data = P39E.top5_bytes(qids, st.top5(scores(members), cand_ids))
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    rp = os.path.join(od, "apply_report.json")
    if os.path.exists(rp):
        os.replace(rp, rp + ".stale")                  # an old report never vouches for the new top5
    with open(out + ".tmp", "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(out + ".tmp", out)
    r29 = [r.split(b",") for r in data29.split(b"\n")[1:] if r]
    r68 = [r.split(b",") for r in data.split(b"\n")[1:] if r]
    rep = {"format": APPLY_FORMAT, "pool": a.pool, "version": ev["version"], "members": members,
           "weights": {m: WEIGHTS[m] for m in members}, "w_E": WE_ENS, "cfg_ens": ce, "member_vectors": used,
           "provenance": prov, "src_csv_sha256": MF.csv_shas(a.src, a.pool),
           "v29_top5_equals_prereg67_apply": True, "questions": len(qids), "candidates": len(cand_ids),
           "games": len(items), "k_counts": {str(x): int((ks == x).sum()) for x in (1, 2, 3, 5)},
           "games_without_P1": int((~t["usable"]).sum()), "questions_without_P1": int((~np.asarray(t["has"])).sum()),
           "eval_sha256": ev_sha, "e_pools_use_number": ev["e_pools_use_number"],
           "top5_sha256": L.sha256_file(out),
           "rows_differ_from_v29": int(sum(x != y for x, y in zip(r68, r29))),
           "top1_differs_from_v29": int(sum(x[1] != y[1] for x, y in zip(r68, r29))),
           "peak_rss_mb": P63.peak_mb(), "seconds": round(time.time() - t0, 1)}
    write_json_atomic(rp, rep)
    print(json.dumps(rep, indent=1))
    return 0


def apply_report_problems(out, ev, ev_sha):
    """Pure-ish (files only): both pools' apply_report.json vs the adopted eval and the top5.csv now on disk."""
    p = []
    for pool in POOLS:
        rp, top = os.path.join(out, pool, "apply_report.json"), os.path.join(out, pool, "top5.csv")
        if not (os.path.exists(rp) and os.path.exists(top)):
            p.append("%s: apply_report.json / top5.csv missing" % pool)
            continue
        r = read_json(rp)
        for k, want in (("format", APPLY_FORMAT), ("pool", pool), ("version", ev["version"]),
                        ("members", ev["members"]), ("eval_sha256", ev_sha),
                        ("e_pools_use_number", ev["e_pools_use_number"]),
                        ("top5_sha256", L.sha256_file(top)), ("v29_top5_equals_prereg67_apply", True)):
            if r.get(k) != want:
                p.append("%s: apply_report %s %r != %r" % (pool, k, r.get(k), want))
    return p


def v29_compare(mine_bytes, v29_bytes):
    mine, v29 = P63.lf(mine_bytes).split(b"\n"), P63.lf(v29_bytes).split(b"\n")
    orig = lambda rows: [r for r in rows if r.startswith((b"pub_", b"priv_"))]  # noqa: E731
    new = lambda rows: [r for r in rows if r.startswith((b"pub2_", b"priv2_"))]  # noqa: E731
    t1 = {r.split(b",")[0]: r.split(b",")[1] for r in new(v29)}
    rep = {"original_800_identical_to_v29": orig(mine) == orig(v29) and len(orig(mine)) == 800,
           "header_identical_to_v29": mine[0] == v29[0],
           "new_top1_changed_vs_v29": {
               p: sum(1 for r in new(mine) if r.startswith(p.encode()) and t1.get(r.split(b",")[0]) != r.split(b",")[1])
               for p in ("pub2_", "priv2_")}}
    return rep


def cmd_assemble(a):
    """NOTES.md 12: the final CSV appears (atomic replace) only after every check passed."""
    final = os.path.join(a.out, "submission_player.csv")
    final_rep = os.path.join(a.out, "assemble_report.json")
    for f in (final, final_rep):
        if os.path.exists(f):
            os.replace(f, f + ".stale")
    ev, ev_sha = adopted_eval()
    prob = apply_report_problems(a.out, ev, ev_sha)
    if prob:
        print(json.dumps({"ok": False, "problems": prob}, indent=1))
        return 1
    stage = os.path.join(a.out, "_assemble_stage")
    shutil.rmtree(stage, ignore_errors=True)
    for pool in POOLS:
        os.makedirs(os.path.join(stage, pool))
        shutil.copyfile(os.path.join(a.out, pool, "top5.csv"), os.path.join(stage, pool, "top5.csv"))
        if L.sha256_file(os.path.join(stage, pool, "top5.csv")) != \
                read_json(os.path.join(a.out, pool, "apply_report.json"))["top5_sha256"]:
            print(json.dumps({"ok": False, "problems": ["%s top5 changed while copying" % pool]}))
            return 1
    rc = E43.cmd_assemble(argparse.Namespace(v1=a.v1, src=a.src, out=stage))
    staged = os.path.join(stage, "submission_player.csv")
    rep = read_json(os.path.join(stage, "assemble_report.json"))
    with open(staged, "rb") as f, open(os.path.join(V29_APPLY, "submission_player.csv"), "rb") as g:
        rep.update(v29_compare(f.read(), g.read()))
    rep["rows_1600"] = rep["rows"] == 1600
    rep["apply_reports"] = {p: L.sha256_file(os.path.join(a.out, p, "apply_report.json")) for p in POOLS}
    rep["eval_sha256"] = ev_sha
    ok = rc == 0 and rep["ok"] and rep["original_800_identical_to_v29"] and rep["rows_1600"] \
        and rep["header_identical_to_v29"]
    rep["published"] = bool(ok)
    print(json.dumps({x: rep.get(x) for x in ("ok", "rows", "sha256", "original_800_identical_to_v29",
                                              "new_top1_changed_vs_v29", "published")}, indent=1))
    if not ok:
        write_json_atomic(os.path.join(stage, "assemble_report.json"), rep)
        return 1
    write_json_atomic(final_rep, rep)
    os.replace(staged, final)
    shutil.rmtree(stage, ignore_errors=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seam")
    s.add_argument("--part", required=True, choices=["pseudo", "e"])
    c = sub.add_parser("check-member")
    c.add_argument("--name", required=True, choices=list(NEW))
    sub.add_parser("version")
    sub.add_parser("e-use")
    j = sub.add_parser("judge")
    j.add_argument("--e-use", type=int, required=True)
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", default=P49.V1_SUBMISSION)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd == "e-use":
        u = e_uses_so_far()
        print(json.dumps(u))
        print(u["count"] + 1)
        return 0
    if a.cmd not in ("assemble", "version"):
        BK.polite()
        BK.wait_for_ram(need_mb=RAM_MIN_MB)
        P63._WD["s"] = P63.rss_watchdog()
    return {"seam": cmd_seam, "check-member": cmd_check_member, "version": cmd_version, "judge": cmd_judge,
            "apply": cmd_apply, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
