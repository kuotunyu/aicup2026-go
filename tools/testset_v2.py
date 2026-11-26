"""PREREG_27 (docs/PREREG_27_TESTSET_V2.md): the official test set v2 (reopened 2026-11-11 11:00). Stdlib only.

The registered pipelines run on v2 unchanged; this tool only (1) keeps the v1 test-day outputs (git mv into
results/archive_testset_v1/, the upload log entries into 'voided_uploads', the data folders into
data/archive_testset_v1/), (2) describes the downloaded v2 files against v1 and the hard-coded structure constants,
(3) for competition-2 branches K / R0 (rank files byte-identical to v1) writes the ingest source: v1's normalised
rank files plus the downloaded competition-1 files, and (4) builds a synthetic v2 download for rehearsals.

    PY tools/testset_v2.py plan     --branch K|R0|RR                    the archive plan as JSON (changes nothing)
    PY tools/testset_v2.py archive  --branch K|R0|RR [--skip-data]      PREREG_27 1: git mv + log + data (no commit)
    PY tools/testset_v2.py probe    --src DIR_OR_ZIP [--src ...] [--v1-dir DIR] [--out PATH]
    PY tools/testset_v2.py rank-src --src DIR_OR_ZIP [--src ...] --dest DIR [--v1-dir DIR]
    PY tools/testset_v2.py fake     --root S --scenario same|grow|grow-player [--extra N] [--swap N]
    PY tools/testset_v2.py compare  --v1 FILE --v2 FILE     record only (PREREG_27 6): rows of the same question id

Branches (PREREG_27 2): K = competition 2 untouched; R0 = competition 2 re-uploads 25 as is (its v1 entries stay in
'uploads' with a 'void' field); RR = competition 2 reruns (its v1 entries move to 'voided_uploads'). Competition 1
always reruns. archive needs tag testset-v1-final to be an ancestor of HEAD, every planned file tracked, clean and
byte-identical to the tag's blob; it refuses an existing archive target. It never commits.

Never opens, lists or hashes anything under data/local: every path this tool touches is a fixed name or a path the
caller gives, and a caller path inside data/local (compared as text, without touching that folder) is refused.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

REGISTRATION = "docs/PREREG_27_TESTSET_V2.md"
TAG = "testset-v1-final"
BASE = "f21ff1e74cc3166279174cc691df43c6968c552c"     # main's last commit before 2026-11-04 11:00 +08:00
ARCH = "results/archive_testset_v1"
DATA_ARCH = "data/archive_testset_v1"
LOG_REL = "results/official_scores.json"
BRANCHES = ("K", "R0", "RR")
NOTICE = ("AICUP 2026 測試集更新及競賽系統暫停與重新開放說明 (AIdea, competition 1 page): uploads paused "
          "2026-11-06 11:00, leaderboard cleared, reopen 2026-11-11 11:00 with original + new questions")
NOTICE_AT = "2026-11-06T11:00:00+08:00"

P_PREFIXES = ("results/test02/player/", "results/test02/unpack_player.json", "results/test02/checks/player/",
              "results/test06/", "results/test10/", "results/test21/")
S_PREFIXES = ("results/test02/checks/built/", "results/test02/checks/bundle_build.json",
              "results/test02/checks/root_props/", "results/test02/checks/t0_tutorial_diff.txt",
              "results/test02/checks/trailing_pass/")
DATA_P = ("data/test", "data/test06", "data/test_rootprops", "data/test_pass", "data/colab_bundle/C10_test_v1.zip",
          "data/colab_upload/test02")
DATA_R = ("data/test07", "data/test08", "data/colab_upload/test04")

V1_NAMES = {"rank_prediction_test_public.csv": ("rank", "public", "q"),
            "rank_prediction_test_private.csv": ("rank", "private", "q"),
            "player_identification_test_public.csv": ("player", "public", "q"),
            "player_identification_test_private.csv": ("player", "private", "q"),
            "player_identification_candidates_public.csv": ("player", "public", "c"),
            "player_identification_candidates_private.csv": ("player", "private", "c")}
RANK_NAMES = ("rank_prediction_test_public.csv", "rank_prediction_test_private.csv")
# the structure constants the v1 tools hard-code (t02_checks / t02_test_bundle / check_submission / notebooks)
V1_STRUCT = {"questions_per_file": 400, "candidates_per_pool": 400, "games_per_candidate": 100,
             "k_min": 5, "k_max": 20, "sgf_columns": 20,
             "q_prefix": {"public": "pub_", "private": "priv_"},
             "p_prefix": {"public": "pub_p_", "private": "priv_p_"},
             "g_prefix": {"public": "pub_", "private": "priv_"}}
HARD_CODED_AT = ["tools/t02_checks.py N_QUESTIONS / N_CANDIDATES / GAMES_PER_CANDIDATE / prefixes (T1.2, T1.3 k 5..20)",
                 "tools/t02_test_bundle.py N_QUESTIONS / GAMES_PER_PLAYER / RANK_FILES / PLAYER_*_FILES",
                 "tools/check_submission.py OFFICIAL_QUESTIONS / OFFICIAL_PLAYERS / OFFICIAL_GAMES_PER_PLAYER / 800 rows",
                 "tools/official_log.py FILE_NAMES / QUESTION_FILES / CANDIDATE_FILES",
                 "colab/build_test02_rank_nb.py N_TEST_QUESTIONS, colab/build_test02_player_nb.py preflight 400/40,000",
                 "NOT a structure constant to change: tools/t06_player.py N_OFFICIAL (06-ON 'pools_400_by_400' is the registered "
                 "literal of PREREG_06 5.3; PREREG_27 7 keeps it)"]
_NEEDLE = "ans" + "wers"


class V2Error(SystemExit):
    def __init__(self, msg):
        super().__init__(f"testset_v2: {msg}")


# ---------------------------------------------------------------- small helpers
def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm(p):
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))


def refuse_local(path):
    """Text comparison only: never touches data/local itself."""
    a, loc = norm(path), norm(os.path.join(REPO, "data", "local"))
    if a == loc or a.startswith(loc + os.sep) or _NEEDLE in os.path.basename(a):
        raise V2Error(f"refusing {path!r}: data/local and label files are never read")
    return path


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(refuse_local(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args, cwd=None, check=True, text=True):
    r = subprocess.run(["git", "-C", cwd or REPO] + list(args), capture_output=True, text=text, timeout=600)
    if check and r.returncode != 0:
        raise V2Error(f"git {' '.join(args)} failed: {(r.stderr or '').strip()[:400]}")
    return r


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- plan / archive (PREREG_27 1, 2)
R_PREFIXES = ("results/test02/rank/", "results/test02/unpack_rank.json", "results/test02/checks/rank/",
              "results/test02/failed/", "results/test04/", "results/test07/", "results/test08/", "results/test09/",
              "results/test14/", "results/test19/", "results/test25/")
# gitignored v1 Colab result zips (not in git; kept with the data, review finding 4)
ZIPS = {"P": ("results/test02/test02_player_results.zip",),
        "R": ("results/test02/test02_rank_results.zip", "results/test04/test04_rank_results.zip")}
ORDER = {"K": 0, "R0": 1, "RR": 2}
VOID_BY_BRANCH = {"K": {"player": "move", "rank": None}, "R0": {"player": "move", "rank": "mark"},
                  "RR": {"player": "move", "rank": "move"}}


def group_of(rel):
    for g, pres in (("P", P_PREFIXES), ("S", S_PREFIXES), ("R", R_PREFIXES)):
        if any(rel == p or rel.startswith(p) for p in pres):
            return g
    raise V2Error(f"{rel}: a v1 output in no group (P / S / R); extend the prefixes under a PREREG_27 revision")


def groups_for(branch):
    if branch not in BRANCHES:
        raise V2Error(f"--branch must be one of {BRANCHES}")
    return ("P", "S", "R") if branch == "RR" else ("P", "S")


def v1_outputs(repo=REPO, tag=TAG, base=BASE):
    """Files under results/test* added between BASE and the tag (the v1 test-day outputs)."""
    r = git("diff", "--diff-filter=A", "--name-only", "-z", base, tag, "--", "results/", cwd=repo)
    out = sorted(p for p in r.stdout.split("\0") if p and re.match(r"results/test[0-9]", p))
    other = git("diff", "--diff-filter=MDRT", "--name-only", "-z", base, tag, "--", "results/", cwd=repo)
    changed = sorted(p for p in other.stdout.split("\0") if p and re.match(r"results/test[0-9]", p))
    if changed:   # a pre-11/04 file changed on test day: it is not a plain v1 output; stop rather than guess
        raise V2Error(f"files under results/test* were modified (not added) after {base[:7]}: {changed[:5]}")
    return out


def data_for(groups):
    return list(DATA_P) + list(ZIPS["P"]) + ((list(DATA_R) + list(ZIPS["R"])) if "R" in groups else [])


def build_plan(branch, repo=REPO, tag=TAG, base=BASE):
    groups = groups_for(branch)
    files = v1_outputs(repo, tag, base)
    by = defaultdict(list)
    for f in files:
        by[group_of(f)].append(f)
    move = sorted(f for g in groups for f in by[g])
    return {"format": "c10 testset_v2 archive plan v1", "registration": REGISTRATION, "branch": branch, "tag": tag,
            "base": base, "groups": list(groups), "counts": {g: len(by[g]) for g in ("P", "S", "R")},
            "files": move, "data": data_for(groups), "void": dict(VOID_BY_BRANCH[branch]), "archive_dir": ARCH,
            "data_archive_dir": DATA_ARCH}


def void_log(log, void, when=None):
    """PREREG_27 1: v1 entries of the rerun competitions move to 'voided_uploads' ('move') or keep their place with
    a 'void' field ('mark'); nothing is deleted; an entry already marked keeps its first 'void' record when it moves
    later (R0 -> RR). Returns the ids touched per action."""
    rec = {"by": "organizer", "notice": NOTICE, "notice_at": NOTICE_AT, "registration": REGISTRATION,
           "voided_at": when or now_iso()}
    keep, moved, marked = [], [], []
    for e in log.get("uploads", []):
        act = void.get(e.get("comp"))
        if act == "move":
            e2 = dict(e)
            e2.setdefault("void", rec)
            log.setdefault("voided_uploads", []).append(e2)
            moved.append(e["id"])
        else:
            if act == "mark" and "void" not in e:
                e["void"] = rec
                marked.append(e["id"])
            keep.append(e)
    log["uploads"] = keep
    for comp, act in void.items():
        if act == "move" and (log.get("final_choice") or {}).get(comp):
            log.setdefault("voided_final_choice", {})[comp] = log["final_choice"][comp]
            log["final_choice"][comp] = None
    return {"moved": moved, "marked": marked}


def tag_blob(repo, tag, rel):
    r = git("rev-parse", f"{tag}:{rel}", cwd=repo, check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def head_blob(repo, rel):
    r = git("rev-parse", f"HEAD:{rel}", cwd=repo, check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def cmd_plan(a):
    print(json.dumps(build_plan(a.branch), indent=1, ensure_ascii=False))
    return 0


def data_dst(rel):
    parts = rel.split("/")
    return "/".join([DATA_ARCH] + (parts[1:] if parts[0] == "data" else parts))


def hash_tree(repo, rel):
    src = os.path.join(repo, *rel.split("/"))
    files = []
    if os.path.isdir(src):
        for d, dirs, fs in os.walk(src):
            dirs.sort()
            for f in sorted(fs):
                q = os.path.join(d, f)
                files.append({"path": os.path.relpath(q, repo).replace(os.sep, "/"), "bytes": os.path.getsize(q),
                              "sha256": sha256_file(q)})
    else:
        files.append({"path": rel, "bytes": os.path.getsize(src), "sha256": sha256_file(src)})
    return files


def archive(branch, repo=REPO, skip_data=False, tag=TAG, base=BASE, log=print, when=None, _fail_after=None):
    """PREREG_27 1. A later, wider branch may follow an earlier one (K -> R0 -> RR; competition 2 cleared after
    11/11): only what the earlier stages did not do is done. Order: every check, then the record (status
    in_progress, with the data manifest), the data renames, git mv, the log; a failure in the renames or in git mv
    undoes what this run moved and leaves the record at status failed. Everything is staged; nothing is committed."""
    plan = build_plan(branch, repo, tag, base)
    if git("merge-base", "--is-ancestor", tag, "HEAD", cwd=repo, check=False).returncode != 0:
        raise V2Error(f"tag {tag} is not an ancestor of HEAD")
    arch_abs = os.path.join(repo, *ARCH.split("/"))
    rec_path = os.path.join(arch_abs, "ARCHIVE.json")
    state = json.load(open(rec_path, encoding="utf-8")) if os.path.exists(rec_path) else None
    done_groups, done_data, done_void = set(), set(), {"player": None, "rank": None}
    if state:
        bad = [s_ for s_ in state.get("stages", []) if s_.get("status") != "complete"]
        if bad:
            raise V2Error(f"{ARCH}/ARCHIVE.json has a stage with status {bad[-1].get('status')!r}: undo it "
                          "(docs/TESTDAY_27_V2.md, recovery) before running again")
        last = state["stages"][-1]["branch"]
        if ORDER[branch] <= ORDER[last]:
            raise V2Error(f"the v1 archive is already at branch {last}; {branch} adds nothing")
        for s_ in state["stages"]:
            done_groups |= set(s_["groups"])
            done_data |= {d["from"] for d in s_.get("data_rows", [])}
            for c, act in s_["void"].items():
                if act:
                    done_void[c] = act
        if not skip_data and any(s_.get("data_skipped") for s_ in state["stages"]):
            raise V2Error("an earlier stage skipped the data; keep --skip-data in this clone")
    files = [f for f in plan["files"] if group_of(f) not in done_groups]
    void = {c: (act if act != done_void.get(c) else None) for c, act in plan["void"].items()}
    # content check on exactly what moves (and the log): no local change, no untracked file in those folders
    dirty = git("diff", "--name-only", "HEAD", "--", LOG_REL, *files, cwd=repo).stdout.strip()
    dirs = sorted({f.rsplit("/", 1)[0] for f in files})
    extra = git("ls-files", "--others", "--exclude-standard", "--", *dirs, cwd=repo).stdout.strip() if dirs else ""
    if dirty or extra:
        raise V2Error(f"uncommitted changes in what would move; commit first:\n{(dirty + chr(10) + extra)[:600]}")
    staged = git("diff", "--cached", "--name-only", cwd=repo).stdout.strip()
    if staged:
        raise V2Error(f"the index already has staged changes; commit or unstage them first:\n{staged[:400]}")
    rows = []
    for rel in files:
        tb, hb = tag_blob(repo, tag, rel), head_blob(repo, rel)
        if tb is None or hb is None or tb != hb:
            raise V2Error(f"{rel}: blob at HEAD {hb} differs from {tag} {tb}; the v1 output changed after the tag")
        dst = f"{ARCH}/{rel}"
        if os.path.exists(os.path.join(repo, *dst.split("/"))):
            raise V2Error(f"{dst} exists")
        rows.append({"from": rel, "to": dst, "blob": tb})
    data_rows = []
    if not skip_data:
        for rel in plan["data"]:
            if rel in done_data:
                continue
            src = os.path.join(repo, *rel.split("/"))
            refuse_local(src)
            dst = os.path.join(repo, *data_dst(rel).split("/"))
            if os.path.exists(dst):
                raise V2Error(f"{dst} exists")
            if not os.path.exists(src):
                data_rows.append({"from": rel, "present": False})
                continue
            data_rows.append({"from": rel, "to": data_dst(rel), "present": True, "files": hash_tree(repo, rel)})
    log_abs = os.path.join(repo, *LOG_REL.split("/"))
    raw = open(log_abs, "rb").read()
    lg = json.loads(raw.decode("utf-8"))
    touched = void_log(lg, void, when)
    tag_commit = git("rev-parse", f"{tag}^{{commit}}", cwd=repo).stdout.strip()
    stage = {"branch": branch, "status": "in_progress", "made_at": when or now_iso(),
             "groups": [g for g in plan["groups"] if g not in done_groups], "counts": plan["counts"],
             "moved": rows, "void": void, "log": touched, "data_skipped": bool(skip_data), "data_rows": data_rows}
    if state is None:
        state = {"format": "c10 testset_v2 archive v2", "registration": REGISTRATION, "tag": tag,
                 "tag_commit": tag_commit, "base": base, "log_copy": f"{ARCH}/official_scores.json",
                 "log_copy_sha256": sha256_bytes(raw), "stages": []}
        os.makedirs(arch_abs, exist_ok=True)
        with open(os.path.join(arch_abs, "official_scores.json"), "wb") as f:
            f.write(raw)                 # the v1 log, byte for byte (first stage only)
    state["stages"].append(stage)
    write_json(rec_path, state)
    moved_data, moved_git = [], []
    try:
        for d in data_rows:               # the step most likely to fail on Windows (open handles): first
            if d.get("present"):
                src = os.path.join(repo, *d["from"].split("/"))
                dst = os.path.join(repo, *d["to"].split("/"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.rename(src, dst)
                moved_data.append((src, dst))
        for r in rows:
            if _fail_after is not None and len(moved_git) >= _fail_after:
                raise OSError("injected failure (test)")
            os.makedirs(os.path.dirname(os.path.join(repo, *r["to"].split("/"))), exist_ok=True)
            git("mv", "--", r["from"], r["to"], cwd=repo)
            moved_git.append(r)
    except BaseException as e:
        for r in reversed(moved_git):
            git("mv", "--", r["to"], r["from"], cwd=repo, check=False)
        for src, dst in reversed(moved_data):
            try:
                os.rename(dst, src)
            except OSError:
                pass
        stage["status"] = "failed"
        stage["error"] = str(e)[:500]
        stage["undone"] = {"git": len(moved_git), "data": len(moved_data)}
        write_json(rec_path, state)
        raise V2Error(f"archive failed and was undone ({len(moved_git)} git moves, {len(moved_data)} data moves): "
                      f"{e}; see {ARCH}/ARCHIVE.json")
    for r in rows:                       # folders the move left empty (git does not track folders); results/ only
        d = os.path.dirname(os.path.join(repo, *r["from"].split("/")))
        stop = os.path.join(repo, "results")
        while norm(d).startswith(norm(stop) + os.sep) and os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
            d = os.path.dirname(d)
    with open(log_abs + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(lg, f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(log_abs + ".tmp", log_abs)
    stage["status"] = "complete"
    write_json(rec_path, state)
    adds = [LOG_REL, f"{ARCH}/ARCHIVE.json", f"{ARCH}/official_scores.json"]
    git("add", "--", *adds, cwd=repo)
    got = git("diff", "--cached", "--name-status", cwd=repo).stdout.splitlines()
    want_r = {(r["from"], r["to"]) for r in rows}
    odd = [x for x in got if not (x.split("\t")[0].startswith("R") and tuple(x.split("\t")[1:3]) in want_r)
           and x.split("\t")[-1] not in adds]
    log(f"archived {len(rows)} files ({', '.join(stage['groups']) or 'none'}) into {ARCH}/; log: moved "
        f"{touched['moved']}, marked {touched['marked']}; data "
        f"{'skipped' if skip_data else str(len(moved_data)) + ' moved'}; staged {len(got)} entries")
    if odd:
        log(f"WARNING: staged entries outside the archive: {odd[:10]}")
    log(f"next: git commit -m \"PREREG_27 archive v1 ({branch})\"   (everything is staged; nothing else)")
    stage["staged_unexpected"] = odd
    return stage


def cmd_archive(a):
    st = archive(a.branch, skip_data=a.skip_data)
    return 1 if st.get("staged_unexpected") else 0


# ---------------------------------------------------------------- probe (PREREG_27 6, 7)
def read_sources(srcs):
    """{basename: bytes} of every .csv in the given folders (recursive) or zips; duplicates must be identical."""
    found, where = {}, {}

    def add(name, b, origin):
        if _NEEDLE in name.lower():
            raise V2Error(f"refusing {origin}")
        if name in found and found[name] != b:
            raise V2Error(f"{name}: two different files ({where[name]}, {origin})")
        found[name], where[name] = b, origin

    for s in srcs:
        refuse_local(s)
        if os.path.isdir(s):
            for d, dirs, fs in os.walk(s):
                dirs.sort()
                for f in sorted(fs):
                    p = os.path.join(d, f)
                    if f.startswith("._") or "__MACOSX" in d:
                        continue
                    if f.lower().endswith(".csv"):
                        add(f, open(p, "rb").read(), p)
                    elif f.lower().endswith(".zip"):
                        for n, b in zip_csvs(p):
                            add(n, b, f"{p}:{n}")
        elif s.lower().endswith(".zip"):
            for n, b in zip_csvs(s):
                add(n, b, f"{s}:{n}")
        elif s.lower().endswith(".csv"):
            add(os.path.basename(s), open(s, "rb").read(), s)
        else:
            raise V2Error(f"--src {s}: not a folder, .zip or .csv")
    return found, where


def zip_csvs(path):
    with zipfile.ZipFile(path) as z:
        for m in sorted(z.namelist()):
            if m.lower().endswith(".csv") and "__MACOSX" not in m and not os.path.basename(m).startswith("._"):
                yield os.path.basename(m), z.read(m)


def id_pattern(ids):
    pats = Counter(re.sub(r"\d", "9", i) for i in ids)
    return dict(pats.most_common(5))


def parse_csv(b):
    csv.field_size_limit(10 ** 9)
    text = b.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    return (rows[0], rows[1:]) if rows else ([], [])


ROOT_RE = re.compile(r"^\s*\(\s*;((?:[A-Za-z]+(?:\[[^\]]*\])+)*)")
PROP_RE = re.compile(r"([A-Za-z]+)((?:\[[^\]]*\])+)")


def root_props(sgf):
    m = ROOT_RE.match(sgf or "")
    return [p for p, _ in PROP_RE.findall(m.group(1))] if m else []


def describe(name, b):
    head, rows = parse_csv(b)
    d = {"bytes": len(b), "sha256": sha256_bytes(b), "rows": len(rows), "header": head[:4] + (["..."] if len(head) > 4
                                                                                             else []),
         "columns": len(head), "crlf": b.count(b"\r\n"), "lf": b.count(b"\n"), "bom": b[:3] == b"\xef\xbb\xbf"}
    roots = Counter()
    if head[:2] == ["question_id", "num_games"]:
        d["kind"] = "questions"
        nsgf = sum(1 for h in head if h.startswith("sgf_"))
        d["sgf_columns"] = nsgf
        ks = Counter()
        cols = Counter()
        for r in rows:
            try:
                k = int(r[1])
            except (ValueError, IndexError):
                k = None
            ks[k] += 1
            for i in range(min(k or 0, nsgf)):
                roots[",".join(root_props(r[2 + i]))] += 1
                if 2 + nsgf + i < len(r):
                    cols[r[2 + nsgf + i]] += 1
        d["k_counts"] = {str(k): v for k, v in sorted(ks.items(), key=lambda x: (x[0] is None, x[0] or 0))}
        d["colours"] = dict(cols)
        d["id_patterns"] = id_pattern(r[0] for r in rows if r)
        d["ids"] = [r[0] for r in rows if r]
        d["duplicate_ids"] = len(d["ids"]) - len(set(d["ids"]))
    elif head[:2] == ["game_id", "player_id"]:
        d["kind"] = "candidates"
        per = Counter(r[1] for r in rows if len(r) > 1)
        d["players"] = len(per)
        d["games_per_player"] = dict(Counter(per.values()))
        d["colours"] = dict(Counter(r[3] for r in rows if len(r) > 3))
        d["player_id_patterns"] = id_pattern(per)
        d["game_id_patterns"] = id_pattern(r[0] for r in rows if r)
        d["ids"] = sorted(per)
        gids = [r[0] for r in rows if r]
        d["duplicate_ids"] = len(gids) - len(set(gids))
        for r in rows:
            if len(r) > 2:
                roots[",".join(root_props(r[2]))] += 1
    else:
        d["kind"] = "unknown"
    d["root_props"] = dict(roots.most_common(8))
    return d, rows


def row_map(rows):
    return {r[0]: r for r in rows if r}


def default_v1_dir():
    for rel in (f"{DATA_ARCH}/test_rootprops/original", "data/test_rootprops/original"):
        p = os.path.join(REPO, *rel.split("/"))
        if os.path.isdir(p):
            return p
    return None


def probe(srcs, v1_dir=None):
    found, where = read_sources(srcs)
    v1_dir = v1_dir or default_v1_dir()
    files, needed = {}, []
    v1 = {}
    if v1_dir:
        refuse_local(v1_dir)
        for n in V1_NAMES:
            p = os.path.join(v1_dir, n)
            if os.path.isfile(p):
                v1[n] = open(p, "rb").read()
    names = sorted(found)
    missing = sorted(set(V1_NAMES) - set(names))
    if set(names) - set(V1_NAMES):    # a new name is a structure change; a missing one may be an incomplete download
        needed.append({"what": "file names", "v1": sorted(V1_NAMES), "v2": names, "missing": missing,
                       "new": sorted(set(names) - set(V1_NAMES))})
    for n in names:
        d, rows = describe(n, found[n])
        d["source"] = where[n]
        role = V1_NAMES.get(n)
        if n in v1:
            vd, vrows = describe(n, v1[n])
            d["v1_byte_identical"] = found[n] == v1[n]
            m2, m1 = row_map(rows), row_map(vrows)
            same = sum(1 for k in m1 if k in m2 and m2[k] == m1[k])
            same_after_root = sum(1 for k in m1 if k in m2 and m2[k] != m1[k] and
                                  [strip_root(x) for x in m2[k]] == [strip_root(x) for x in m1[k]])
            d["v1_compare"] = {"v1_rows": len(vrows), "v1_ids_kept": sum(1 for k in m1 if k in m2),
                               "rows_identical": same, "rows_identical_after_root_props": same_after_root,
                               "rows_changed": sum(1 for k in m1 if k in m2) - same - same_after_root,
                               "v1_ids_missing": sum(1 for k in m1 if k not in m2),
                               "new_ids": sum(1 for k in m2 if k not in m1),
                               "v1_ids_first": bool(rows) and [r[0] for r in rows[:len(vrows)]] == [r[0] for r in vrows]}
        if role:
            comp, pool, kind = role
            if kind == "q":
                if d.get("rows") != V1_STRUCT["questions_per_file"]:
                    needed.append({"what": "questions per file", "file": n, "v1": 400, "v2": d.get("rows")})
                ks = [int(k) for k in d.get("k_counts", {}) if k not in ("None",)]
                if ks and (min(ks) < V1_STRUCT["k_min"] or max(ks) > V1_STRUCT["k_max"]) or "None" in d.get("k_counts",
                                                                                                          {}):
                    needed.append({"what": "num_games range", "file": n, "v1": [5, 20], "v2": d.get("k_counts")})
                if d.get("sgf_columns") != V1_STRUCT["sgf_columns"]:
                    needed.append({"what": "sgf columns", "file": n, "v1": 20, "v2": d.get("sgf_columns")})
                pre = V1_STRUCT["q_prefix"][pool]
                bad = [i for i in d.get("ids", []) if not i.startswith(pre)]
                if bad:
                    needed.append({"what": "question id prefix", "file": n, "v1": pre, "examples": bad[:3]})
            else:
                if d.get("players") != V1_STRUCT["candidates_per_pool"]:
                    needed.append({"what": "candidates per pool", "file": n, "v1": 400, "v2": d.get("players")})
                if set(d.get("games_per_player", {})) != {V1_STRUCT["games_per_candidate"]}:
                    needed.append({"what": "games per candidate", "file": n, "v1": 100,
                                   "v2": d.get("games_per_player")})
                pre = V1_STRUCT["p_prefix"][pool]
                bad = [i for i in d.get("ids", []) if not i.startswith(pre)]
                if bad:
                    needed.append({"what": "player id prefix", "file": n, "v1": pre, "examples": bad[:3]})
        if d.get("duplicate_ids"):
            needed.append({"what": "duplicate ids", "file": n, "v2": d["duplicate_ids"]})
        d.pop("ids", None)
        files[n] = d
    rank_given = all(n in found for n in RANK_NAMES)
    rank_same = (all(n in v1 and found[n] == v1[n] for n in RANK_NAMES) if v1 and rank_given else None)
    player_same = (all(n in found and n in v1 and found[n] == v1[n] for n in V1_NAMES if V1_NAMES[n][0] == "player")
                   if v1 else None)
    pq = [files.get(n, {}) for n in V1_NAMES if V1_NAMES[n] in (("player", "public", "q"),
                                                                ("player", "private", "q"))]
    pc = [files.get(n, {}) for n in V1_NAMES if V1_NAMES[n][0] == "player" and V1_NAMES[n][2] == "c"]
    return {"format": "c10 testset_v2 probe v1", "registration": REGISTRATION, "made_at": now_iso(),
            "v1_dir": v1_dir.replace(os.sep, "/") if v1_dir else None, "files": files,
            "structure_matches_v1": not needed and not missing, "needed_changes": needed, "missing_files": missing, "hard_coded_at": HARD_CODED_AT,
            "rank_bytes_identical_to_v1": rank_same, "player_bytes_identical_to_v1": player_same,
            "player_pools_square": [q.get("rows") == c.get("players") for q, c in zip(pq, pc)] if pq and pc else None,
            "rank_files_given": rank_given,
            "competition_2_branch_by_bytes": ("not decidable: the two rank files are not in --src" if not rank_given
                                              else None if rank_same is None else
                                              "K or R0 (page decides: no notice and leaderboard kept -> K)" if rank_same
                                              else "RR")}


def strip_root(cell):
    """The cell with root properties other than SZ removed (HD etc.; record only, for the v1 comparison)."""
    m = ROOT_RE.match(cell or "")
    if not m or not m.group(1):
        return cell
    keep = "".join(p + v for p, v in PROP_RE.findall(m.group(1)) if p in ("SZ", "B", "W", "AB", "AW", "AE", "HA",
                                                                            "PL"))
    return cell[:m.start(1)] + keep + cell[m.end(1):]


def cmd_probe(a):
    rep = probe(a.src, a.v1_dir)
    out = a.out or os.path.join(REPO, "results", "testset_v2", "probe.json")
    refuse_local(out)
    write_json(out, rep)
    print(f"wrote {out}")
    for n, d in rep["files"].items():
        extra = (f" players {d.get('players')} games/player {d.get('games_per_player')}" if d.get("kind") == "candidates"
                 else f" k {d.get('k_counts')}")
        cmp_ = d.get("v1_compare") or {}
        print(f"  {n}: {d['rows']} rows{extra}; v1 identical {d.get('v1_byte_identical')}; kept {cmp_.get('v1_ids_kept')}"
              f" same {cmp_.get('rows_identical')} (+{cmp_.get('rows_identical_after_root_props')} after root) "
              f"new {cmp_.get('new_ids')}")
    print(f"structure matches v1: {rep['structure_matches_v1']}; rank bytes identical: "
          f"{rep['rank_bytes_identical_to_v1']}; competition 2 by bytes: {rep['competition_2_branch_by_bytes']}")
    for c in rep["needed_changes"]:
        print(f"  NEEDED: {json.dumps(c, ensure_ascii=False)[:300]}")
    if rep["missing_files"]:
        print(f"  MISSING (download again or add --src): {rep['missing_files']}")
    return 0 if rep["structure_matches_v1"] else 4


# ---------------------------------------------------------------- rank-src (PREREG_27 2, K / R0)
def rank_src(srcs, dest, v1_dir=None):
    """The ingest source of branches K / R0 (PREREG_27 2): an empty --dest gets v1's normalised rank files (after the
    downloaded rank files proved byte-identical to v1's originals) and the downloaded competition-1 files as they are."""
    found, _ = read_sources(srcs)
    v1_dir = v1_dir or default_v1_dir()
    if not v1_dir:
        raise V2Error("v1 original folder not found (data/archive_testset_v1/test_rootprops/original)")
    normalised = os.path.join(os.path.dirname(v1_dir), "normalised")
    if not all(os.path.isfile(os.path.join(normalised, n)) for n in RANK_NAMES):
        raise V2Error(f"{normalised}: the v1 normalised rank files are missing")
    unknown = sorted(set(found) - set(V1_NAMES))
    if unknown:
        raise V2Error(f"--src has files that are not the 6 v1 names {unknown}: the format changed (PREREG_27 7); "
                      "run probe")
    refuse_local(dest)
    if os.path.exists(dest) and os.listdir(dest):
        raise V2Error(f"--dest {dest} must be new or empty")
    rec = {}
    for n in RANK_NAMES:
        if n not in found:
            raise V2Error(f"{n} is not in --src")
        v1b = open(os.path.join(v1_dir, n), "rb").read()
        if found[n] != v1b:
            raise V2Error(f"{n}: the download differs from v1 ({sha256_bytes(found[n])[:12]} vs "
                          f"{sha256_bytes(v1b)[:12]}): competition 2 changed, branch RR (PREREG_27 2)")
        rec[n] = {"download_sha256": sha256_bytes(found[n]), "v1_normalised_sha256": sha256_file(os.path.join(
            normalised, n))}
    for n in V1_NAMES:
        if n not in RANK_NAMES and n not in found:
            raise V2Error(f"{n} is not in --src")
    os.makedirs(dest, exist_ok=True)
    for n in RANK_NAMES:
        shutil.copyfile(os.path.join(normalised, n), os.path.join(dest, n))
    for n in V1_NAMES:               # competition 1: the download's bytes as they are
        if n not in RANK_NAMES:
            with open(os.path.join(dest, n), "wb") as f:
                f.write(found[n])
            rec[n] = {"download_sha256": sha256_bytes(found[n]), "copied": "download bytes"}
    return rec


def cmd_rank_src(a):
    rec = rank_src(a.src, a.dest, a.v1_dir)
    out = a.record or os.path.join(REPO, "results", "testset_v2", "rank_src.json")
    refuse_local(out)
    write_json(out, {"format": "c10 testset_v2 rank-src v1", "registration": REGISTRATION, "made_at": now_iso(),
                     "src": [x.replace(os.sep, "/") for x in a.src], "dest": a.dest.replace(os.sep, "/"),
                     "v1_dir": (a.v1_dir or default_v1_dir() or "").replace(os.sep, "/"), "files": rec})
    print(f"wrote {out}")
    for n, r in rec.items():
        print(f"{n}: download {r['download_sha256'][:12]}" + (f" = v1 original; copied v1 normalised "
              f"{r['v1_normalised_sha256'][:12]}" if "v1_normalised_sha256" in r else " copied as is") + f" -> {a.dest}")
    return 0


# ---------------------------------------------------------------- fake (rehearsal only)
FAKE_SALT = "c10-testset-v2-fake"


def _h(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def with_hd(sgf):
    return sgf.replace("(;SZ[19]", "(;SZ[19]HD[0]", 1) if sgf.startswith("(;SZ[19]") else sgf


def q_line(qid, games, colour):
    k = len(games)
    cells = [qid, str(k)] + games + [""] * (20 - k) + [colour] * k + [""] * (20 - k)
    return ",".join(cells).encode("utf-8") + b"\r\n"


def fake(root, scenario, extra=100, swap=50, v1_dir=None, rows_fit=None, log=print):
    if scenario not in ("same", "grow", "grow-player"):
        raise V2Error("--scenario must be same, grow or grow-player")
    r = norm(root)
    if r == norm(REPO) or r.startswith(norm(REPO) + os.sep):
        raise V2Error("--root must be outside the repository")
    v1_dir = v1_dir or default_v1_dir()
    if not v1_dir:
        raise V2Error("v1 original folder not found")
    dl = os.path.join(root, "download")
    if os.path.exists(dl):
        raise V2Error(f"{dl} exists")
    if rows_fit is None:
        import c10data   # training players only (split v1, part fit); local_dir=None: data/local is not opened
        rows_fit = c10data.load_rows(os.path.join(REPO, "data", "training"),
                                     os.path.join(REPO, "results", "split_v1", "players_split.csv"), part="fit",
                                     local_dir=None)
    by_player = defaultdict(lambda: {"B": [], "W": []})
    for row in rows_fit:
        if row["color"] in ("B", "W") and "," not in row["sgf_content"] and '"' not in row["sgf_content"]:
            by_player[row["player_id"]][row["color"]].append(row["sgf_content"])
    pids = sorted(by_player, key=lambda p: _h(FAKE_SALT, p))
    used = set()

    def take(n_black, need_colour=None, k=None):
        for p in pids:
            if p in used:
                continue
            g = by_player[p]
            if need_colour is None and len(g["B"]) >= n_black:
                used.add(p)
                return p, g["B"]
            if need_colour and len(g[need_colour]) >= k:
                used.add(p)
                return p, g[need_colour]
        raise V2Error("not enough fit players")

    out = {}
    v1 = {n: open(os.path.join(v1_dir, n), "rb").read() for n in V1_NAMES}
    rec = {"format": "c10 testset_v2 fake v1", "scenario": scenario, "extra": extra, "swap": swap, "v1_dir": v1_dir,
           "note": "rehearsal only: training (fit) games with synthesized ids; never uploaded", "pools": {}}
    ks = (5, 10, 15, 20)
    for pool, pre in (("public", "pub"), ("private", "priv")):
        qn, cn = f"player_identification_test_{pool}.csv", f"player_identification_candidates_{pool}.csv"
        qlines = v1[qn].split(b"\r\n")[:-1]
        clines = v1[cn].split(b"\r\n")[:-1]
        if scenario == "same":
            # replace the last `swap` candidates (100 rows each) and the last `swap` question rows; counts unchanged
            n_c = (len(clines) - 1) // 100
            keep_c = clines[:1 + (n_c - swap) * 100]
            new_c, new_q = [], []
            gid = int(keep_c[-1].split(b",")[0].decode().rsplit("_", 1)[1])
            for j in range(swap):
                p, g = take(120)
                pid = f"{pre}_p_{n_c - swap + j + 1:04d}"
                for s in g[:100]:
                    gid += 1
                    new_c.append(f"{pre}_g_{gid:07d},{pid},{with_hd(s)},B".encode() + b"\r\n")
                k = ks[j % 4]
                qrow = qlines[len(qlines) - swap + j]
                qid = qrow.split(b",")[0].decode()
                new_q.append(q_line(qid, [with_hd(s) for s in g[100:100 + k]], "B"))
            out[cn] = b"\r\n".join(keep_c) + b"\r\n" + b"".join(new_c)
            out[qn] = b"\r\n".join(qlines[:len(qlines) - swap]) + b"\r\n" + b"".join(new_q)
        else:
            n_c = (len(clines) - 1) // 100
            n_q = len(qlines) - 1
            gid = int(clines[-1].split(b",")[0].decode().rsplit("_", 1)[1])
            new_c, new_q = [], []
            for j in range(extra):
                p, g = take(120)
                pid = f"{pre}_p_{n_c + j + 1:04d}"
                for s in g[:100]:
                    gid += 1
                    new_c.append(f"{pre}_g_{gid:07d},{pid},{with_hd(s)},B".encode() + b"\r\n")
                k = ks[j % 4]
                new_q.append(q_line(f"{pre}_q_{n_q + j + 1:04d}", [with_hd(s) for s in g[100:100 + k]], "B"))
            out[cn] = v1[cn] + b"".join(new_c)
            out[qn] = v1[qn] + b"".join(new_q)
        rec["pools"][pool] = {"player_questions": out[qn].count(b"\r\n") - 1,
                              "candidates": (out[cn].count(b"\r\n") - 1) // 100}
    for n in RANK_NAMES:
        if scenario == "grow":
            pool, pre = ("public", "pub") if "public" in n else ("private", "priv")
            n_q = v1[n].count(b"\r\n") - 1
            new_q = []
            for j in range(extra):
                k = ks[j % 4]
                colour = "B" if int(_h(FAKE_SALT, n, j), 16) % 2 else "W"
                p, g = take(0, colour, k)
                new_q.append(q_line(f"{pre}_q_{n_q + j + 1:04d}", [with_hd(s) for s in g[:k]], colour))
            out[n] = v1[n] + b"".join(new_q)
        else:
            out[n] = v1[n]
    os.makedirs(dl)
    for n, b in out.items():
        with open(os.path.join(dl, n), "wb") as f:
            f.write(b)
    rec["files"] = {n: {"sha256": sha256_bytes(b), "rows": b.count(b"\r\n") - 1} for n, b in sorted(out.items())}
    write_json(os.path.join(root, "FAKE_V2.json"), rec)
    log(f"wrote {dl} ({scenario}): " + ", ".join(f"{n} {v['rows']}" for n, v in rec["files"].items()))
    return rec


def cmd_fake(a):
    fake(a.root, a.scenario, a.extra, a.swap, a.v1_dir)
    return 0


# ---------------------------------------------------------------- compare (PREREG_27 6, record only)
def compare(v1_path, v2_path):
    """Two submission files (question_id first): on the question ids of both, how many rows are identical and how
    many differ in the first prediction column. Record only; never a gate."""
    out = {}
    for tag, p in (("v1", v1_path), ("v2", v2_path)):
        refuse_local(p)
        head, rows = parse_csv(open(p, "rb").read())
        out[tag] = (head, {r[0]: r[1:] for r in rows if r})
    (h1, a), (h2, b) = out["v1"], out["v2"]
    common = [k for k in a if k in b]
    return {"format": "c10 testset_v2 compare v1", "v1": v1_path.replace(os.sep, "/"),
            "v2": v2_path.replace(os.sep, "/"), "header_equal": h1 == h2, "v1_rows": len(a), "v2_rows": len(b),
            "common_ids": len(common), "rows_identical": sum(1 for k in common if a[k] == b[k]),
            "first_column_differs": sum(1 for k in common if a[k][:1] != b[k][:1]),
            "by_pool": {pre: sum(1 for k in common if k.startswith(pre) and a[k][:1] != b[k][:1])
                        for pre in ("pub_", "priv_")}}


def cmd_compare(a):
    print(json.dumps(compare(a.v1, a.v2), indent=1, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--branch", required=True, choices=BRANCHES)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("archive")
    p.add_argument("--branch", required=True, choices=BRANCHES)
    p.add_argument("--skip-data", action="store_true", help="git and log part only (rehearsal clones)")
    p.set_defaults(fn=cmd_archive)
    p = sub.add_parser("probe")
    p.add_argument("--src", action="append", required=True)
    p.add_argument("--v1-dir", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("rank-src")
    p.add_argument("--src", action="append", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--record", default=None, help="default results/testset_v2/rank_src.json")
    p.add_argument("--v1-dir", default=None)
    p.set_defaults(fn=cmd_rank_src)
    p = sub.add_parser("fake")
    p.add_argument("--root", required=True)
    p.add_argument("--scenario", required=True, choices=("same", "grow", "grow-player"))
    p.add_argument("--extra", type=int, default=100)
    p.add_argument("--swap", type=int, default=50)
    p.add_argument("--v1-dir", default=None)
    p.set_defaults(fn=cmd_fake)
    p = sub.add_parser("compare")
    p.add_argument("--v1", required=True)
    p.add_argument("--v2", required=True)
    p.set_defaults(fn=cmd_compare)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
