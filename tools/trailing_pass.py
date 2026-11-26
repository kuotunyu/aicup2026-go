"""PREREG_02 rev 9 (docs/PREREG_02_REV9_TRAILING_PASS.md): remove the trailing passes of the official test files.

T1.4 (PREREG_02 9.6) counts any pass (B[] / B[tt] / W[] / W[tt]) as a violation; more than 1% of a file's games is
FAIL_OPENING. Training games have no pass, but official test games that ended by counting may keep one or two passes
after the last real move. When the raw T1 of a competition is FAIL_OPENING only because of such games, this tool
writes a copy of the official files with exactly those pass nodes removed (byte-level minimal edit), keeps the
original files read-only, and records the mapping. The unchanged tools then take the copy as the raw files:

    PY tools/trailing_pass.py prepare [--root S] [--rerun --reason TEXT]
        needs data/test/raw (6 official files + SOURCE.json of the first ingest) and the committed full raw T1 of both
        competitions (results/test02/checks/{rank,player}/t1.json), and no build yet (data/test/MANIFEST.json and
        results/test02/checks/built absent). Writes
          data/test_pass/original/   byte copies of the 6 files + SOURCE.json, read-only
          data/test_pass/stripped/   the 6 official names only (copies; trailing passes removed where rev 9 applies)
          results/test02/checks/trailing_pass/prepare.json, raw_t1_<comp>.json (byte copies of the raw t1.json)
        exit 0: rev 9 applies to at least one competition (next: ingest --src data/test_pass/stripped --rerun, then
        t1 --rerun for both competitions, then verify); 2: not needed / not applicable (original rule: go on with the
        recorded T1 verdict; only the records prepare.json and raw_t1_<comp>.json are written, no data); 1: refusal.
    PY tools/trailing_pass.py verify [--root S] [--rerun --reason TEXT]
        data/test/raw equals the copy, SOURCE.json names the copy, the originals are unchanged, both t1.json record the
        current raw files, the applied competition is PASS / NORMALIZED as predicted, the other keeps its verdict.
        Writes results/test02/checks/trailing_pass/verify.json. exit 0 passed, 1 not passed / refusal.

Modes: real (default; repo paths, this file and the inputs committed) or --root S (a scratch tree with the repo's
layout, outside the repository; no git checks). Never reads or writes data/local; opens only the six official file
names, SOURCE.json and the t1.json / own records. Imports the frozen tools/t02_checks.py read-only (sgf_check, t1_check,
file names and small helpers); nothing in it is changed.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t02_checks as tc  # noqa: E402  (frozen; imported read-only)

FORMAT_PREPARE = "c10-rev9 trailing pass prepare v1"
FORMAT_VERIFY = "c10-rev9 trailing pass verify v1"
SOURCE_FORMAT = "c10-test02 source v1"
REGISTRATION = "docs/PREREG_02_REV9_TRAILING_PASS.md"
COMPS = ("rank", "player")
COMP_FILES = {c: [(pool, kind, name) for pool, kind, name in tc.FILES[c]] for c in COMPS}
ALL_NAMES = [n for c in COMPS for _p, _k, n in COMP_FILES[c]]
T1_OK = ("PASS", "NORMALIZED")
BOM = b"\xef\xbb\xbf"

# a pass node exactly as T1.4 sees it (t02_checks.PASS_RE / sgf_props): ';' [BW] '[' ('' | 'tt') ']', with the
# whitespace T1.4 tolerates; the run of pass nodes must reach the closing ')' of the game
_PASS_NODE = r";\s*[BW]\s*\[(?:tt)?\]"
TAIL_RE = re.compile(r"%s(?:\s*%s)*(?=\s*\)\s*\Z)" % (_PASS_NODE, _PASS_NODE))
PASS_NODE_RE = re.compile(_PASS_NODE)
LAST_MOVE_RE = re.compile(r";\s*[BW]\s*\[[a-s]{2}\]\s*\Z")
ANY_MOVE_RE = re.compile(r"(?<![A-Za-z])[BW]\s*\[[a-s]{2}\]")       # record only: where the real moves are
ANY_PASS_RE = re.compile(r"(?<![A-Za-z])[BW]\s*\[(?:tt)?\]")        # and the passes (mid-game vs after the last move)


class Refuse(SystemExit):
    def __init__(self, msg):
        super().__init__(f"trailing_pass: {msg}")


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_bytes(path):
    with open(tc.refuse_forbidden(path), "rb") as f:
        return f.read()


# ================================================================ one game
def classify_game(sgf):
    """(status, stripped sgf or None, detail). status: 'no_pass', 'trailing' (rev 9 removes the tail),
    'pass_with_other_reason' (a pass and another T1.4 reason), 'mid_game_pass' (a pass before the last real move),
    'pass_not_trailing_other' (only passes after the last real move, but not a bare tail of pass nodes before ')':
    a pass node with other properties, a variation, no closing ')'), 'pass_parse_error'. None of these but
    'trailing' is ever edited."""
    g = tc.sgf_check(sgf)
    if not g["pass"]:
        return "no_pass", None, None
    if g["parse_error"] is not None:
        return "pass_parse_error", None, {"reasons": g["reasons"], "parse_error": g["parse_error"]}
    if g["reasons"] != ["pass"]:
        return "pass_with_other_reason", None, {"reasons": g["reasons"]}
    moves = [x.start() for x in ANY_MOVE_RE.finditer(sgf)]
    if moves and any(x.start() < moves[-1] for x in ANY_PASS_RE.finditer(sgf)):
        return "mid_game_pass", None, {"reasons": g["reasons"]}
    m = TAIL_RE.search(sgf)
    if m is None or not LAST_MOVE_RE.search(sgf[:m.start()]):
        return "pass_not_trailing_other", None, {"reasons": g["reasons"]}
    new = sgf[:m.start()] + sgf[m.end():]
    h = tc.sgf_check(new)
    if h["reasons"] or h["pass"] or h["moves"] != g["moves"] or h["parse_error"] is not None:
        return "pass_not_trailing_other", None, {"reasons": g["reasons"], "after": h["reasons"]}
    return "trailing", new, {"start": m.start(), "end": m.end(), "removed": sgf[m.start():m.end()],
                             "pass_nodes": len(PASS_NODE_RE.findall(sgf[m.start():m.end()]))}


# ================================================================ CSV cells with their character spans
def scan_cells(text):
    """[[(value, start, end, quoted)]] per record, the way csv.reader (excel dialect, newline='') splits text;
    (start, end) is the span of the value characters (inside the quotes for a quoted field). A blank line is []."""
    out, i, n = [], 0, len(text)
    while i < n:
        rec = []
        while True:
            if i < n and text[i] == '"':
                j = i + 1
                while True:
                    k = text.find('"', j)
                    if k < 0:
                        raise ValueError("unterminated quoted field at %d" % i)
                    if k + 1 < n and text[k + 1] == '"':
                        j = k + 2
                        continue
                    break
                rec.append((text[i + 1:k].replace('""', '"'), i + 1, k, True))
                i = k + 1
                if i < n and text[i] not in ',\r\n':
                    raise ValueError("text after a closing quote at %d" % i)
            else:
                k = i
                while k < n and text[k] not in ',\r\n':
                    k += 1
                rec.append((text[i:k], i, k, False))
                i = k
            if i < n and text[i] == ",":
                i += 1
                continue
            if text.startswith("\r\n", i):
                i += 2
            elif i < n:
                i += 1
            break
        if len(rec) == 1 and rec[0][0] == "" and not rec[0][3]:
            rec = []
        out.append(rec)
    return out


def _csv_rows(text):
    csv.field_size_limit(10 ** 8)
    return list(csv.reader(io.StringIO(text, newline="")))


# ================================================================ one file
def analyse_file(path, kind):
    """Per-file classification and the stripped bytes (None when nothing is removed). Uses the same game set as T1.4
    (t02_checks.file_games: question slots 1..num_games, every candidate row)."""
    raw = read_bytes(path)
    t = tc.Table(path)
    rec = {"file": os.path.basename(path), "sha256_before": sha256_bytes(raw), "bytes_before": len(raw),
           "games": 0, "counts": {}, "edited": [], "not_edited_examples": [], "scan_problem": None}
    if t.error or t.bad_width:
        rec["scan_problem"] = t.error or "rows with another width: %s" % t.bad_width[:5]
        return rec, None
    bom = raw.startswith(BOM)
    text = raw[len(BOM):].decode("utf-8") if bom else raw.decode("utf-8")
    try:
        cells = scan_cells(text)
    except ValueError as e:
        rec["scan_problem"] = "scan: %s" % e
        return rec, None
    if [[c[0] for c in r] for r in cells] != _csv_rows(text):
        rec["scan_problem"] = "the character scan and csv.reader disagree"
        return rec, None
    records = [r for r in cells if r]
    if not records:
        rec["scan_problem"] = "no header"
        return rec, None
    header = [c[0] for c in records[0]]
    body = records[1:]
    if len(body) != len(t.rows) or any([c[0] for c in r] != row for r, row in zip(body, t.rows)):
        rec["scan_problem"] = "rows differ from t02_checks.Table"
        return rec, None
    col = {}
    for j, c in enumerate(header):
        col.setdefault(c, j)
    # the T1.4 game set with each game's cell
    slots = []
    if kind == "questions":
        for r in body:
            ng = r[col["num_games"]][0] if "num_games" in col else ""
            nmax = int(ng) if re.fullmatch(r"[0-9]+", ng or "") else None
            qid = r[col["question_id"]][0] if "question_id" in col else ""
            for i in range(1, tc.N_SLOTS + 1):
                j = col.get("sgf_%d" % i)
                if j is None:
                    continue
                s = r[j][0]
                if (nmax is not None and i <= nmax) or (nmax is None and s.strip()):
                    slots.append(("%s/sgf_%d" % (qid, i), r[j]))
    else:
        j = col.get("sgf_content")
        g = col.get("game_id")
        if j is not None:
            for r in body:
                slots.append((r[g][0] if g is not None else "?", r[j]))
    check = [s for _lab, (s, _a, _b, _q) in slots]
    want = [gm[3] for gm in tc.file_games(t, kind)]
    if check != want:
        rec["scan_problem"] = "game set differs from t02_checks.file_games"
        return rec, None
    counts = {k: 0 for k in ("no_pass", "trailing", "trailing_unusual_cell", "pass_with_other_reason",
                             "mid_game_pass", "pass_not_trailing_other", "pass_parse_error")}
    counts["pass_nodes_removed"] = 0
    counts["max_pass_nodes_in_one_tail"] = 0
    edits = []
    cursor = [0, len(BOM) if bom else 0]          # (char position, its byte offset in the file): offsets increase

    def byte_at(c):
        if c < cursor[0]:
            cursor[0], cursor[1] = 0, len(BOM) if bom else 0
        cursor[1] += len(text[cursor[0]:c].encode("utf-8"))
        cursor[0] = c
        return cursor[1]

    for label, (value, a, b, quoted) in slots:
        status, new, detail = classify_game(value)
        if status == "trailing" and text[a:b] != value:
            status = "trailing_unusual_cell"       # escaped quotes inside the cell: not edited (rev 9 section 2)
        counts[status] += 1
        if status == "trailing":
            edits.append((a + detail["start"], a + detail["end"]))
            counts["pass_nodes_removed"] += detail["pass_nodes"]
            counts["max_pass_nodes_in_one_tail"] = max(counts["max_pass_nodes_in_one_tail"], detail["pass_nodes"])
            rec["edited"].append({"game": label, "removed": detail["removed"], "pass_nodes": detail["pass_nodes"],
                                  "cell_chars": [detail["start"], detail["end"]],
                                  "file_bytes": [byte_at(a + detail["start"]), byte_at(a + detail["end"])]})
        elif status != "no_pass" and len(rec["not_edited_examples"]) < 20:
            rec["not_edited_examples"].append({"game": label, "status": status, "detail": detail})
    rec["games"] = len(slots)
    rec["counts"] = counts
    if not edits:
        return rec, None
    parts, pos = [], 0
    for a, b in sorted(edits):
        if a < pos:
            raise Refuse(f"{path}: overlapping edits")
        parts.append(text[pos:a])
        pos = b
    parts.append(text[pos:])
    new_text = "".join(parts)
    new_raw = (BOM if bom else b"") + new_text.encode("utf-8")
    # self-checks: only the listed spans are gone, every other cell is unchanged, the edited cells lost only the tail
    back, pos2 = [], 0
    for a, b in sorted(edits):
        back.append(text[pos2:a] + text[a:b])
        pos2 = b
    back.append(text[pos2:])
    if "".join(back) != text:
        raise Refuse(f"{path}: reconstruction check failed")
    old_rows, new_rows = _csv_rows(text), _csv_rows(new_text)
    if len(old_rows) != len(new_rows) or any(len(x) != len(y) for x, y in zip(old_rows, new_rows)):
        raise Refuse(f"{path}: the copy has another CSV shape")
    changed = sum(x != y for orow, nrow in zip(old_rows, new_rows) for x, y in zip(orow, nrow))
    if changed != len(edits):
        raise Refuse(f"{path}: {changed} cells changed, {len(edits)} expected")
    rec["bytes_removed"] = len(raw) - len(new_raw)
    return rec, new_raw


# ================================================================ paths, guards, git
def paths(root):
    base = root or REPO
    p = {"raw": os.path.join(base, "data", "test", "raw"), "tests": os.path.join(base, "data", "test"),
         "pass": os.path.join(base, "data", "test_pass"), "checks": os.path.join(base, "results", "test02", "checks")}
    p["original"] = os.path.join(p["pass"], "original")
    p["stripped"] = os.path.join(p["pass"], "stripped")
    p["rec"] = os.path.join(p["checks"], "trailing_pass")
    p["built"] = os.path.join(p["checks"], "built")
    p["manifest"] = os.path.join(p["tests"], "MANIFEST.json")
    return p


def check_root(root):
    if root is None:
        return
    if tc.inside(root, REPO):
        raise Refuse(f"--root {root!r} is inside the repository; scratch trees live outside it")


def guard_paths(p):
    for key in ("pass", "rec"):
        if tc.inside(p[key], os.path.join(REPO, "data", "local")):
            raise Refuse(f"{p[key]!r} is inside data/local")


def committed_clean(paths_, what):
    probs = tc.not_committed_clean(paths_)
    if probs:
        raise Refuse(f"real mode needs {what} committed and clean: " + "; ".join(probs))


def rel_or_abs(path, root):
    base = root or REPO
    try:
        r = os.path.relpath(os.path.abspath(path), os.path.abspath(base))
    except ValueError:
        return os.path.abspath(path)
    return r.replace(os.sep, "/") if not r.startswith("..") else os.path.abspath(path)


def read_t1(p, comp, real):
    path = os.path.join(p["checks"], comp, "t1.json")
    if not os.path.isfile(path):
        raise Refuse(f"{path} is missing: run the full T1 of both competitions first (docs/TESTDAY.md step 1)")
    t1 = tc.read_json(path)
    mode = (t1.get("provenance") or {}).get("mode")
    if t1.get("complete") is not True or t1.get("stage") != "raw":
        raise Refuse(f"{path}: complete={t1.get('complete')!r}, stage {t1.get('stage')!r}; rev 9 needs the full T1 "
                     f"(T1.1-T1.5) of the raw files")
    if real and mode != "real":
        raise Refuse(f"{path}: mode {mode!r}; real mode needs the real T1")
    if real:
        committed_clean([path], f"{rel_or_abs(path, None)} (commit the raw T1 first, whatever its verdict)")
    return path, t1


def recorded_shas(t1):
    return {n: (v or {}).get("sha256") for n, v in (t1.get("files") or {}).items()}


def require_fresh(targets, args):
    exist = [t for t in targets if os.path.exists(t)]
    if exist and not args.rerun:
        raise Refuse(f"outputs exist: {exist[:4]}; replace them only with --rerun --reason TEXT")
    if args.rerun and not args.reason:
        raise Refuse("--rerun needs --reason TEXT")
    return bool(exist)


def write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    if sha256_bytes(read_bytes(path)) != sha256_bytes(data):
        raise Refuse(f"{path}: SHA-256 changed while writing")


def make_writable_tree(d):
    for r, _dirs, files in os.walk(d):
        for f in files:
            os.chmod(os.path.join(r, f), stat.S_IREAD | stat.S_IWRITE)


# ================================================================ prepare
def decide(comp, t1, analyses):
    """(decision, reason) before the copy check: not_needed / not_applicable / candidate."""
    v = t1.get("verdict")
    if v == "FAIL_STRUCT":
        return "not_applicable", "raw T1 verdict FAIL_STRUCT: stop and write a revision (PREREG_02 9.6)"
    if v != "FAIL_OPENING":
        return "not_needed", f"raw T1 verdict {v}"
    subs = t1.get("subchecks") or {}
    bad = [s for s in ("T1.1", "T1.2", "T1.3", "T1.5") if (subs.get(s) or {}).get("verdict") == "FAIL"]
    if bad:
        return "not_applicable", f"raw T1 has {bad} FAIL (rev 9 section 3 item 2)"
    if (subs.get("T1.4") or {}).get("verdict") != "FAIL":
        return "not_applicable", "raw T1.4 is not FAIL"
    probs = [a["file"] + ": " + a["scan_problem"] for a in analyses if a["scan_problem"]]
    n_trailing = sum(a["counts"].get("trailing", 0) for a in analyses)
    if n_trailing == 0:
        return "not_applicable", "no trailing-pass game" + (f" (scan problems: {probs})" if probs else "")
    return "candidate", f"{n_trailing} trailing-pass games" + (f"; files not edited: {probs}" if probs else "")


def cmd_prepare(args):
    real = args.root is None
    check_root(args.root)
    p = paths(args.root)
    guard_paths(p)
    if real:
        committed_clean([os.path.abspath(__file__), os.path.join(REPO, *REGISTRATION.split("/")),
                         os.path.join(HERE, "t02_checks.py")],
                        "tools/trailing_pass.py, tools/t02_checks.py and the rev 9 registration")
    for n in ALL_NAMES + ["SOURCE.json"]:
        if not os.path.isfile(os.path.join(p["raw"], n)):
            raise Refuse(f"{os.path.join(p['raw'], n)} missing: ingest the AIdea download first (docs/TESTDAY.md 1)")
    src = tc.read_json(os.path.join(p["raw"], "SOURCE.json"))
    if src.get("format") != SOURCE_FORMAT:
        raise Refuse("data/test/raw/SOURCE.json is not an ingest record")
    if any(tc.same_path(s, p["stripped"]) or tc.inside(s, p["pass"]) for s in src.get("sources") or []):
        raise Refuse("data/test/raw already holds the rev 9 copy (SOURCE.json names data/test_pass); prepare runs on "
                     "the first ingest of the AIdea download only")
    if os.path.exists(p["manifest"]) or os.path.exists(p["built"]):
        raise Refuse("the bundle was already built (data/test/MANIFEST.json or results/test02/checks/built exists): "
                     "rev 9 section 4 item 5 - STOP and report to the main session; do not redo steps")
    raw_sha = {n: tc.sha256_file(os.path.join(p["raw"], n)) for n in ALL_NAMES + ["SOURCE.json"]}
    t1s, t1_paths = {}, {}
    for comp in COMPS:
        t1_paths[comp], t1s[comp] = read_t1(p, comp, real)
        rs = recorded_shas(t1s[comp])
        for _pool, _kind, n in COMP_FILES[comp]:
            if rs.get(n) != raw_sha[n]:
                raise Refuse(f"{t1_paths[comp]} records {str(rs.get(n))[:12]} for {n}, the raw file is "
                             f"{raw_sha[n][:12]}: the T1 is not of these files")
    out_json = os.path.join(p["rec"], "prepare.json")
    targets = [out_json] + [os.path.join(p["rec"], f"raw_t1_{c}.json") for c in COMPS] + [p["pass"]]
    replaced = require_fresh(targets, args)

    analyses, new_bytes = {}, {}
    for comp in COMPS:
        for _pool, kind, n in COMP_FILES[comp]:
            analyses[n], new_bytes[n] = analyse_file(os.path.join(p["raw"], n), kind)
    comps = {}
    for comp in COMPS:
        names = [n for _p, _k, n in COMP_FILES[comp]]
        d, why = decide(comp, t1s[comp], [analyses[n] for n in names])
        comps[comp] = {"raw_t1_verdict": t1s[comp].get("verdict"), "decision": d, "reason": why,
                       "raw_t1": rel_or_abs(t1_paths[comp], args.root), "raw_t1_sha256": tc.sha256_file(t1_paths[comp])}
    # the unchanged T1 on the raw files must agree with the committed one (same tool, same files)
    for comp in COMPS:
        if comps[comp]["decision"] == "not_needed":
            continue
        again = tc.t1_check(comp, p["raw"])
        if again["verdict"] != t1s[comp]["verdict"]:
            raise Refuse(f"t1_check on the raw {comp} files gives {again['verdict']}, the committed t1.json "
                         f"{t1s[comp]['verdict']}")
    # the copy check: the unchanged t1_check (T1.1-T1.5) on the stripped files of each candidate competition
    # --rerun keeps data/test_pass/original when it holds exactly the current raw files (rev 9 4.2: never changed);
    # originals of other files are never deleted here
    if os.path.exists(p["original"]):
        have = {n: (tc.sha256_file(os.path.join(p["original"], n)) if os.path.isfile(os.path.join(p["original"], n))
                    else None) for n in raw_sha}
        if have != raw_sha:
            raise Refuse(f"{p['original']} holds other files than data/test/raw (an earlier prepare of another "
                         f"download?): keep it, move it aside by hand and report to the main session")
    for sub in ("stripped", "check_rank", "check_player"):
        d = os.path.join(p["pass"], sub)
        if os.path.exists(d):
            make_writable_tree(d)
            shutil.rmtree(d)
    for comp in COMPS:
        if comps[comp]["decision"] != "candidate":
            continue
        chk = os.path.join(p["pass"], "check_" + comp)
        for _pool, _kind, n in COMP_FILES[comp]:
            write_bytes(os.path.join(chk, n), new_bytes[n] if new_bytes[n] is not None else
                        read_bytes(os.path.join(p["raw"], n)))
        res = tc.t1_check(comp, chk)
        shutil.rmtree(chk)
        t14 = {n: {k: d.get(k) for k in ("games", "violating_games", "share", "fails")}
               for n, d in (res["subchecks"]["T1.4"]["files"] or {}).items()}
        comps[comp]["copy_t1_predicted"] = {"verdict": res["verdict"], "normalize": res["normalize"],
                                            "subchecks": {s: res["subchecks"][s]["verdict"] for s in tc.T1_SUBS},
                                            "T1.4_files": t14}
        if res["verdict"] in T1_OK:
            comps[comp]["decision"] = "apply"
        else:
            comps[comp]["decision"] = "not_applicable"
            comps[comp]["reason"] += f"; the copy's T1 is still {res['verdict']} (rev 9 section 3 item 4)"
    applied = [c for c in COMPS if comps[c]["decision"] == "apply"]
    files = {}
    for comp in COMPS:
        for pool, kind, n in COMP_FILES[comp]:
            a = analyses[n]
            edit = comp in applied and new_bytes[n] is not None
            after = new_bytes[n] if edit else None
            files[n] = {"comp": comp, "pool": pool, "kind": kind, "games": a["games"], "counts": a["counts"],
                        "scan_problem": a["scan_problem"], "sha256_before": a["sha256_before"],
                        "bytes_before": a["bytes_before"],
                        "sha256_after": sha256_bytes(after) if edit else a["sha256_before"],
                        "bytes_after": len(after) if edit else a["bytes_before"], "edited": edit,
                        "games_stripped": len(a["edited"]) if edit else 0,
                        "pass_nodes_removed": a["counts"].get("pass_nodes_removed", 0) if edit else 0,
                        "stripped_games": a["edited"] if edit else [],
                        "not_edited_examples": a["not_edited_examples"]}
    rec = {"format": FORMAT_PREPARE, "registration": REGISTRATION, "created": now_iso(),
           "mode": "real" if real else "scratch", "root": None if real else os.path.abspath(args.root),
           "decision": "apply" if applied else "not_applicable_or_not_needed", "applied_comps": applied,
           "comps": comps, "files": files,
           "raw": {"dir": rel_or_abs(p["raw"], args.root), "sha256": raw_sha, "source_json": src.get("sources")},
           "original_dir": rel_or_abs(p["original"], args.root) if applied else None,
           "stripped_dir": rel_or_abs(p["stripped"], args.root) if applied else None,
           "stripped_dir_abs": os.path.abspath(p["stripped"]) if applied else None,
           "rerun_reason": args.reason if replaced else None,
           "tool": {"file": "tools/trailing_pass.py", "sha256": tc.sha256_file(os.path.abspath(__file__)),
                    "t02_checks_sha256": tc.sha256_file(os.path.join(HERE, "t02_checks.py")), "git_head": tc.git_head()}}
    os.makedirs(p["rec"], exist_ok=True)
    for comp in COMPS:
        write_bytes(os.path.join(p["rec"], f"raw_t1_{comp}.json"), read_bytes(t1_paths[comp]))
    if applied:
        for n in ALL_NAMES + ["SOURCE.json"]:
            o = os.path.join(p["original"], n)
            if not os.path.isfile(o):
                write_bytes(o, read_bytes(os.path.join(p["raw"], n)))
                os.chmod(o, stat.S_IREAD)
        for n in ALL_NAMES:
            data = new_bytes[n] if files[n]["edited"] else read_bytes(os.path.join(p["raw"], n))
            write_bytes(os.path.join(p["stripped"], n), data)
            if sha256_bytes(data) != files[n]["sha256_after"]:
                raise Refuse(f"{n}: copy SHA-256 mismatch")
        orig = {n: tc.sha256_file(os.path.join(p["original"], n)) for n in ALL_NAMES + ["SOURCE.json"]}
        if orig != raw_sha:
            raise Refuse("the original copies differ from data/test/raw")
    elif os.path.isdir(p["pass"]) and not os.listdir(p["pass"]):
        os.rmdir(p["pass"])
    tc.write_json(out_json, rec)
    for comp in COMPS:
        c = comps[comp]
        print(f"{comp}: raw T1 {c['raw_t1_verdict']} -> {c['decision']} ({c['reason']})"
              + (f"; copy T1 {c['copy_t1_predicted']['verdict']}" if "copy_t1_predicted" in c else ""))
    for n in ALL_NAMES:
        f = files[n]
        if f["edited"] or any(f["counts"].get(k) for k in ("trailing", "trailing_unusual_cell", "mid_game_pass",
                                                             "pass_not_trailing_other", "pass_with_other_reason")):
            print(f"  {n}: games {f['games']}, stripped {f['games_stripped']} ({f['pass_nodes_removed']} pass "
                  f"nodes), counts {f['counts']}")
    print("wrote", out_json)
    if not applied:
        print("rev 9 does not apply: go on with the recorded T1 verdict (docs/TESTDAY.md; FAIL_OPENING keeps its "
              "registered consequence)")
        return 2
    rr = "PREREG_02 rev 9: trailing passes removed (results/test02/checks/trailing_pass/prepare.json)"
    if real:
        print("next (docs/TESTDAY_02_PASS.md): commit results/test02/checks/trailing_pass, then")
        print(f'  PY tools/t02_test_bundle.py ingest --src data/test_pass/stripped --rerun --reason "{rr}"')
        for comp in COMPS:
            print(f'  PY tools/t02_checks.py t1 --comp {comp} --rerun --reason "{rr}"')
        print("  commit results/test02/checks/{rank,player}, then  PY tools/trailing_pass.py verify")
    else:
        print(f"next (scratch): ingest --src {p['stripped']} --root ... --rerun, t1 --dry-run --tests-dir "
              f"{p['raw']} --out {p['checks']} --rerun for both competitions, then verify --root ...")
    return 0


# ================================================================ verify
def tp_analyse(path, kind):
    return analyse_file(path, kind)


def cmd_verify(args):
    real = args.root is None
    check_root(args.root)
    p = paths(args.root)
    guard_paths(p)
    pj = os.path.join(p["rec"], "prepare.json")
    if not os.path.isfile(pj):
        raise Refuse(f"{pj} missing: run prepare first")
    if real:
        committed_clean([os.path.abspath(__file__), pj] + [os.path.join(p["rec"], f"raw_t1_{c}.json") for c in COMPS],
                        "tools/trailing_pass.py and the prepare records")
    prep = tc.read_json(pj)
    if prep.get("format") != FORMAT_PREPARE or not prep.get("applied_comps"):
        raise Refuse(f"{pj}: no competition was applied; nothing to verify")
    out = os.path.join(p["rec"], "verify.json")
    replaced = require_fresh([out], args)
    probs, checks = [], {}
    files = prep["files"]
    for n in ALL_NAMES:
        f = files[n]
        got = {"original": tc.sha256_file(os.path.join(p["original"], n)),
               "stripped": tc.sha256_file(os.path.join(p["stripped"], n)),
               "raw": tc.sha256_file(os.path.join(p["raw"], n))}
        checks[n] = got
        if got["original"] != f["sha256_before"]:
            probs.append(f"{n}: the original copy changed")
        if got["stripped"] != f["sha256_after"]:
            probs.append(f"{n}: the stripped copy changed")
        if got["raw"] != f["sha256_after"]:
            probs.append(f"{n}: data/test/raw is not the rev 9 copy (ingest --src data/test_pass/stripped --rerun)")
    # the copy is re-derived from the originals with the same rule (not only trusted from prepare.json)
    for comp in COMPS:
        for _pool, kind, n in COMP_FILES[comp]:
            f = files[n]
            _a, nb = tp_analyse(os.path.join(p["original"], n), kind)
            want = (sha256_bytes(nb) if (nb is not None and f["edited"]) else f["sha256_before"])
            if want != f["sha256_after"]:
                probs.append(f"{n}: re-deriving the copy from the original gives {want[:12]}, prepare.json "
                             f"{f['sha256_after'][:12]}")
    if tc.sha256_file(os.path.join(p["original"], "SOURCE.json")) != prep["raw"]["sha256"]["SOURCE.json"]:
        probs.append("the original SOURCE.json changed")
    src = tc.read_json(os.path.join(p["raw"], "SOURCE.json"))
    srcs = src.get("sources") or []
    if src.get("format") != SOURCE_FORMAT or len(srcs) != 1 or not tc.same_path(srcs[0], p["stripped"]):
        probs.append(f"data/test/raw/SOURCE.json sources {srcs}: not the single rev 9 copy {p['stripped']}")
    for n in ALL_NAMES:
        if ((src.get("files") or {}).get(n) or {}).get("sha256") != files[n]["sha256_after"]:
            probs.append(f"SOURCE.json records another SHA-256 for {n}")
    t1rep = {}
    for comp in COMPS:
        try:
            path, t1 = read_t1(p, comp, real)
        except Refuse as e:
            probs.append(str(e))
            continue
        rs = recorded_shas(t1)
        c = prep["comps"][comp]
        t1rep[comp] = {"verdict": t1.get("verdict"), "sha256": tc.sha256_file(path),
                       "rerun_reason": (t1.get("provenance") or {}).get("rerun_reason")}
        for _pool, _kind, n in COMP_FILES[comp]:
            if rs.get(n) != files[n]["sha256_after"]:
                probs.append(f"{comp} t1.json records {str(rs.get(n))[:12]} for {n}, not the copy: rerun t1 --comp "
                             f"{comp} --rerun --reason ...")
        if c["decision"] == "apply":
            want = c["copy_t1_predicted"]["verdict"]
            if t1.get("verdict") not in T1_OK or t1.get("verdict") != want:
                probs.append(f"{comp}: T1 of the copy is {t1.get('verdict')}, predicted {want}")
        elif t1.get("verdict") != c["raw_t1_verdict"]:
            probs.append(f"{comp}: T1 verdict {t1.get('verdict')} differs from the raw {c['raw_t1_verdict']}")
    passed = not probs
    rec = {"format": FORMAT_VERIFY, "registration": REGISTRATION, "created": now_iso(),
           "mode": "real" if real else "scratch", "passed": passed, "problems": probs,
           "prepare_sha256": tc.sha256_file(pj), "applied_comps": prep["applied_comps"], "t1": t1rep,
           "sha256": checks, "mapping": {n: {"original": files[n]["sha256_before"], "copy": files[n]["sha256_after"],
                                             "edited": files[n]["edited"]} for n in ALL_NAMES},
           "build_already_present": os.path.exists(p["manifest"]),
           "rerun_reason": args.reason if replaced else None}
    tc.write_json(out, rec)
    print("verify:", "PASSED" if passed else "NOT PASSED")
    for x in probs:
        print("  -", x)
    print("wrote", out)
    if passed:
        print("next: commit results/test02/checks (trailing_pass/verify.json and both t1.json), then docs/TESTDAY.md "
              "step 2 (build without --allow-fail-opening)")
    return 0 if passed else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--root", default=None, help="scratch tree outside the repository (no git checks)")
        s.add_argument("--rerun", action="store_true")
        s.add_argument("--reason", default=None)
    args = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
