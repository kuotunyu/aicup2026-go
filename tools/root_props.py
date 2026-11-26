"""PREREG_02 rev 11 (docs/PREREG_22_ROOT_PROPS.md): remove the root properties that training v1.0 never had.

Training v1.1 (2026-11-02) adds HD[n] to every game's root ("(;SZ[19]HD[2];B[pd];..."); the official Q&A says the root
keeps CA / HD / RE. Every frozen model was built on v1.0, whose root is exactly "(;SZ[19]", and the KataGo chain
(results/prereg07/explore/pilot_humansl/hs_lib.parse_moves) rejects any other root. This tool writes a copy of the
official files with every root property except SZ / B / W and the T1.4 setup properties (AB AW AE HA PL) removed
(byte-level minimal edit; moves and non-root nodes never touched), keeps the originals read-only and records the
mapping. The unchanged tools then take the copy as the raw files (the same mechanism as rev 9, tools/trailing_pass.py):

    PY tools/root_props.py prepare [--root S] [--rerun --reason TEXT]
        needs data/test/raw (6 official files + SOURCE.json), the committed full T1 of both competitions recording
        these files, no build yet; when a T1 is FAIL_OPENING, rev 9 first (its prepare.json; when rev 9 was applied,
        its verify.json PASSED). Writes
          data/test_rootprops/original/     byte copies of the 6 files + SOURCE.json, read-only
          data/test_rootprops/normalised/   the 6 official names (root properties removed where rev 11 applies)
          results/test02/checks/root_props/prepare.json, pre_t1_<comp>.json (byte copies of the current t1.json)
        exit 0: applied to at least one competition (next: ingest --src data/test_rootprops/normalised --rerun, t1
        --rerun for both competitions, verify); 2: not needed / not applicable (records only, no data); 1: refusal.
    PY tools/root_props.py verify [--root S] [--rerun --reason TEXT]
        Writes results/test02/checks/root_props/verify.json. exit 0 passed, 1 not passed / refusal.

Modes: real (default; repo paths, this file and the inputs committed) or --root S (scratch tree outside the repository;
no git checks). Never reads or writes data/local; opens only the six official file names, SOURCE.json, the t1.json and
the rev 9 / own records. Imports the frozen tools/t02_checks.py and tools/trailing_pass.py read-only.
"""

import argparse
import collections
import os
import re
import shutil
import stat
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t02_checks as tc  # noqa: E402  (frozen; read-only)
import trailing_pass as tp  # noqa: E402  (rev 9; read-only: CSV cell scan, small helpers)

FORMAT_PREPARE = "c10-rev11 root props prepare v1"
FORMAT_VERIFY = "c10-rev11 root props verify v1"
SOURCE_FORMAT = tp.SOURCE_FORMAT
REGISTRATION = "docs/PREREG_22_ROOT_PROPS.md"
COMPS = tp.COMPS
COMP_FILES = tp.COMP_FILES
ALL_NAMES = tp.ALL_NAMES
BOM = tp.BOM
KEEP = frozenset(("SZ", "B", "W") + tuple(tc.SETUP_PROPS))
OPEN_RE = re.compile(r"\s*\(\s*;")
UPPER_RE = re.compile(r"[A-Z]+")
REASON = "PREREG_02 rev 11: root properties removed (results/test02/checks/root_props/prepare.json)"


class Refuse(SystemExit):
    def __init__(self, msg):
        super().__init__(f"root_props: {msg}")


# ================================================================ one game
def root_tokens(sgf):
    """[(identifier, values, id_start, end)] of the root node, or None when the text does not open with '(' ';' or
    the root does not parse with the T1.4 tokens (t02_checks.TOKEN_RE). end includes the values' trailing blanks."""
    m = OPEN_RE.match(sgf)
    if m is None:
        return None
    pos, out = m.end(), []
    while True:
        t = tc.TOKEN_RE.match(sgf, pos)
        if t is None:
            return out if not sgf[pos:].strip() else None
        if t.group("bad") is not None:
            return None
        if t.group("p") is not None:
            return out                      # the next node, a variation or the end of the game
        out.append((t.group("id"), tc.VAL_RE.findall(t.group("v")), t.start("id"), t.end()))
        pos = t.end()


def classify_game(sgf):
    """(status, new sgf or None, detail). status: 'plain' (nothing to remove), 'root_props' (rev 11 removes them),
    'root_unparsed' / 'parse_error' / 'check_mismatch' (not edited)."""
    toks = root_tokens(sgf)
    if toks is None:
        g = tc.sgf_check(sgf)
        return ("parse_error" if g["parse_error"] else "root_unparsed"), None, {"reasons": g["reasons"]}
    if any(not UPPER_RE.fullmatch(x[0]) for x in toks):
        return "root_unparsed", None, {"why": "root identifier not in upper case"}
    drop = [x for x in toks if x[0] not in KEEP]
    if not drop:
        return "plain", None, None
    g = tc.sgf_check(sgf)
    if g["parse_error"] is not None:
        return "parse_error", None, {"parse_error": g["parse_error"]}
    root_end = toks[-1][3]
    if any(b > root_end for _k, _v, _a, b in drop):
        return "check_mismatch", None, {"why": "span beyond the root node"}
    parts, pos = [], 0
    for _k, _v, a, b in drop:
        parts.append(sgf[pos:a])
        pos = b
    parts.append(sgf[pos:])
    new = "".join(parts)
    h = tc.sgf_check(new)
    old_props, _e = tc.sgf_props(sgf)
    new_props, e2 = tc.sgf_props(new)
    want = list(old_props)
    for k, v, _a, _b in drop:            # the dropped ones are the first matching (identifier, values) of the root
        want.remove((k, v))
    if (e2 is not None or new_props != want or h["moves"] != g["moves"] or h["reasons"] != g["reasons"]
            or h["setup"] != g["setup"] or h["sz"] != g["sz"] or h["pass"] != g["pass"]):
        return "check_mismatch", None, {"reasons": g["reasons"], "after": h["reasons"]}
    rt = root_tokens(new)
    if rt is None or any(k not in KEEP for k, _v, _a, _b in rt):
        return "check_mismatch", None, {"why": "root still has removable properties"}
    return "root_props", new, {"spans": [[a, b] for _k, _v, a, b in drop],
                               "removed": [sgf[a:b] for _k, _v, a, b in drop],
                               "props": [(k, v) for k, v, _a, _b in drop]}


def plain_form(sgf):
    return tc.FAST_SGF.fullmatch(sgf) is not None


# ================================================================ one file
def analyse_file(path, kind):
    """Per-file record and the normalised bytes (None when nothing is removed). Same cell scan and game set as
    rev 9 (trailing_pass.scan_cells, t02_checks.file_games)."""
    raw = tp.read_bytes(path)
    t = tc.Table(path)
    rec = {"file": os.path.basename(path), "sha256_before": tp.sha256_bytes(raw), "bytes_before": len(raw),
           "games": 0, "counts": {}, "removed_props": {}, "hd_values": {}, "edited": [], "not_edited_examples": [],
           "plain_form_before": 0, "plain_form_after": 0, "scan_problem": None}
    if t.error or t.bad_width:
        rec["scan_problem"] = t.error or "rows with another width: %s" % t.bad_width[:5]
        return rec, None
    bom = raw.startswith(BOM)
    text = raw[len(BOM):].decode("utf-8") if bom else raw.decode("utf-8")
    try:
        cells = tp.scan_cells(text)
    except ValueError as e:
        rec["scan_problem"] = "scan: %s" % e
        return rec, None
    if [[c[0] for c in r] for r in cells] != tp._csv_rows(text):
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
    if [s for _lab, (s, _a, _b, _q) in slots] != [gm[3] for gm in tc.file_games(t, kind)]:
        rec["scan_problem"] = "game set differs from t02_checks.file_games"
        return rec, None
    counts = collections.Counter({k: 0 for k in ("plain", "root_props", "unusual_cell", "root_unparsed",
                                                 "parse_error", "check_mismatch", "non_root_other_props")})
    removed, hd = collections.Counter(), collections.Counter()
    edits = []
    for label, (value, a, b, _quoted) in slots:
        rec["plain_form_before"] += plain_form(value)
        status, new, detail = classify_game(value)
        if status == "root_props" and text[a:b] != value:
            status = "unusual_cell"          # escaped quotes inside the cell: not edited (rev 11 section 2)
        counts[status] += 1
        if status == "root_props":
            for (x, y) in detail["spans"]:
                edits.append((a + x, a + y))
            for k, v in detail["props"]:
                removed[k] += 1
                if k == "HD":
                    hd["|".join(v)] += 1
            rec["plain_form_after"] += plain_form(new)
            if set(tc.sgf_check(new)["other_props"]) - KEEP:
                counts["non_root_other_props"] += 1
            if len(rec["edited"]) < 50:
                rec["edited"].append({"game": label, "removed": detail["removed"], "cell_chars": detail["spans"]})
        else:
            rec["plain_form_after"] += plain_form(value)
            if status == "plain" and set(tc.sgf_check(value)["other_props"]) - KEEP:
                counts["non_root_other_props"] += 1
            if status != "plain" and len(rec["not_edited_examples"]) < 20:
                rec["not_edited_examples"].append({"game": label, "status": status, "detail": detail})
            if status != "plain":
                for k, v, _x, _y in (root_tokens(value) or []):
                    if k not in KEEP:
                        removed["not_removed:" + k] += 1
    rec["games"] = len(slots)
    rec["counts"] = dict(counts)
    rec["games_edited"] = counts["root_props"]
    rec["removed_props"] = dict(sorted(removed.items()))
    rec["hd_values"] = dict(sorted(hd.items()))
    if not edits:
        return rec, None
    parts, pos = [], 0
    for x, y in sorted(edits):
        if x < pos:
            raise Refuse(f"{path}: overlapping edits")
        parts.append(text[pos:x])
        pos = y
    parts.append(text[pos:])
    new_text = "".join(parts)
    new_raw = (BOM if bom else b"") + new_text.encode("utf-8")
    old_rows, new_rows = tp._csv_rows(text), tp._csv_rows(new_text)
    if len(old_rows) != len(new_rows) or any(len(x) != len(y) for x, y in zip(old_rows, new_rows)):
        raise Refuse(f"{path}: the copy has another CSV shape")
    changed = sum(x != y for orow, nrow in zip(old_rows, new_rows) for x, y in zip(orow, nrow))
    if changed != counts["root_props"]:
        raise Refuse(f"{path}: {changed} cells changed, {counts['root_props']} expected")
    rec["bytes_removed"] = len(raw) - len(new_raw)
    return rec, new_raw


# ================================================================ paths and guards
def paths(root):
    p = tp.paths(root)
    base = root or REPO
    p["rp"] = os.path.join(base, "data", "test_rootprops")
    p["original"] = os.path.join(p["rp"], "original")
    p["normalised"] = os.path.join(p["rp"], "normalised")
    p["rec"] = os.path.join(p["checks"], "root_props")
    p["tp_rec"] = os.path.join(p["checks"], "trailing_pass")
    p["tp_stripped"] = os.path.join(base, "data", "test_pass", "stripped")
    return p


def guard_paths(p):
    for key in ("rp", "rec"):
        if tc.inside(p[key], os.path.join(REPO, "data", "local")):
            raise Refuse(f"{p[key]!r} is inside data/local")


def committed_clean(paths_, what):
    probs = tc.not_committed_clean(paths_)
    if probs:
        raise Refuse(f"real mode needs {what} committed and clean: " + "; ".join(probs))


def read_t1(p, comp, real):
    path = os.path.join(p["checks"], comp, "t1.json")
    if not os.path.isfile(path):
        raise Refuse(f"{path} is missing: run the full T1 of both competitions first (docs/TESTDAY.md step 1)")
    t1 = tc.read_json(path)
    if t1.get("complete") is not True or t1.get("stage") != "raw":
        raise Refuse(f"{path}: complete={t1.get('complete')!r}, stage {t1.get('stage')!r}; rev 11 needs the full T1")
    if real and (t1.get("provenance") or {}).get("mode") != "real":
        raise Refuse(f"{path}: real mode needs the real T1")
    if real:
        committed_clean([path], f"{tp.rel_or_abs(path, None)} (commit the T1 first)")
    return path, t1


def t1_profile(res):
    """The parts of a T1 that rev 11 must not change: verdict, normalize, sub-verdicts, T1.4 / T1.5 counts per file."""
    subs = res.get("subchecks") or {}
    return {"verdict": res.get("verdict"), "normalize": res.get("normalize"),
            "subchecks": {s: (subs.get(s) or {}).get("verdict") for s in tc.T1_SUBS},
            "T1.4": {n: [d.get("games"), d.get("violating_games")] for n, d in
                     ((subs.get("T1.4") or {}).get("files") or {}).items()},
            "T1.5": {n: [d.get("games"), d.get("first_move_in_S99")] for n, d in
                     ((subs.get("T1.5") or {}).get("files") or {}).items()}}


def rev9_state(p, src, t1s, raw_sha, real):
    """Refuses unless rev 9 is settled for these very files: FAIL_OPENING needs its prepare.json; rev 9 not applied:
    that prepare ran on the current raw files; rev 9 applied: data/test/raw is its copy and its verify.json PASSED
    for this prepare.json. Returns a short record."""
    sources = src.get("sources") or []
    on_copy = any(tc.same_path(s, p["tp_stripped"]) for s in sources)
    pj, vj = os.path.join(p["tp_rec"], "prepare.json"), os.path.join(p["tp_rec"], "verify.json")
    fo = [c for c in COMPS if t1s[c].get("verdict") == "FAIL_OPENING"]
    if fo and not os.path.isfile(pj):
        raise Refuse(f"T1 of {fo} is FAIL_OPENING: run rev 9 first (docs/TESTDAY_02_PASS.md, trailing_pass.py "
                     f"prepare), then this tool; if rev 9 refuses, report to the main session")
    if on_copy and not os.path.isfile(pj):
        raise Refuse("data/test/raw is the rev 9 copy but trailing_pass/prepare.json is missing")
    rec = {"fail_opening": fo, "raw_is_rev9_copy": on_copy, "rev9_prepare": os.path.isfile(pj),
           "rev9_verify_passed": None}
    if not os.path.isfile(pj):
        return rec
    if real:
        committed_clean([pj], "the rev 9 prepare.json")
    prep = tc.read_json(pj)
    applied = prep.get("applied_comps") or []
    rec["rev9_applied"] = applied
    rec["rev9_prepare_sha256"] = tc.sha256_file(pj)
    if not applied:
        if on_copy:
            raise Refuse("data/test/raw names the rev 9 copy but rev 9 applied nothing")
        if any((prep.get("raw") or {}).get("sha256", {}).get(n) != raw_sha[n] for n in ALL_NAMES):
            raise Refuse("trailing_pass/prepare.json is of other files than the current data/test/raw (stale): rerun "
                         "rev 9 prepare on these files (docs/TESTDAY_02_PASS.md) or report to the main session")
        return rec
    if not on_copy:
        raise Refuse("rev 9 was applied (trailing_pass/prepare.json) but data/test/raw is not its copy: finish "
                     "docs/TESTDAY_02_PASS.md first")
    if any(((prep.get("files") or {}).get(n) or {}).get("sha256_after") != raw_sha[n] for n in ALL_NAMES):
        raise Refuse("data/test/raw differs from the rev 9 copy recorded in trailing_pass/prepare.json")
    if not os.path.isfile(vj):
        raise Refuse("data/test/raw is the rev 9 copy but trailing_pass/verify.json is missing: run "
                     "trailing_pass.py verify first")
    if real:
        committed_clean([vj], "the rev 9 verify.json")
    v = tc.read_json(vj)
    rec["rev9_verify_passed"] = v.get("passed")
    if v.get("passed") is not True or v.get("prepare_sha256") != rec["rev9_prepare_sha256"]:
        raise Refuse("trailing_pass/verify.json did not pass for this prepare.json: finish docs/TESTDAY_02_PASS.md")
    return rec


# ================================================================ prepare
def cmd_prepare(args):
    real = args.root is None
    tp.check_root(args.root)
    p = paths(args.root)
    guard_paths(p)
    if real:
        committed_clean([os.path.abspath(__file__), os.path.join(REPO, *REGISTRATION.split("/")),
                         os.path.join(HERE, "t02_checks.py"), os.path.join(HERE, "trailing_pass.py")],
                        "tools/root_props.py, tools/t02_checks.py, tools/trailing_pass.py and the rev 11 registration")
    for n in ALL_NAMES + ["SOURCE.json"]:
        if not os.path.isfile(os.path.join(p["raw"], n)):
            raise Refuse(f"{os.path.join(p['raw'], n)} missing: ingest the AIdea download first (docs/TESTDAY.md 1)")
    src = tc.read_json(os.path.join(p["raw"], "SOURCE.json"))
    if src.get("format") != SOURCE_FORMAT:
        raise Refuse("data/test/raw/SOURCE.json is not an ingest record")
    if any(tc.same_path(s, p["normalised"]) or tc.inside(s, p["rp"]) for s in src.get("sources") or []):
        raise Refuse("data/test/raw already holds the rev 11 copy (SOURCE.json names data/test_rootprops)")
    if os.path.exists(p["manifest"]) or os.path.exists(p["built"]):
        raise Refuse("the bundle was already built (data/test/MANIFEST.json or results/test02/checks/built exists): "
                     "STOP and report to the main session; do not redo steps")
    raw_sha = {n: tc.sha256_file(os.path.join(p["raw"], n)) for n in ALL_NAMES + ["SOURCE.json"]}
    t1s, t1_paths = {}, {}
    for comp in COMPS:
        t1_paths[comp], t1s[comp] = read_t1(p, comp, real)
        rs = tp.recorded_shas(t1s[comp])
        for _pool, _kind, n in COMP_FILES[comp]:
            if rs.get(n) != raw_sha[n]:
                raise Refuse(f"{t1_paths[comp]} records {str(rs.get(n))[:12]} for {n}, the raw file is "
                             f"{raw_sha[n][:12]}: the T1 is not of these files")
    r9 = rev9_state(p, src, t1s, raw_sha, real)
    out_json = os.path.join(p["rec"], "prepare.json")
    targets = [out_json] + [os.path.join(p["rec"], f"pre_t1_{c}.json") for c in COMPS] + [p["rp"]]
    replaced = tp.require_fresh(targets, args)

    analyses, new_bytes = {}, {}
    for comp in COMPS:
        for _pool, kind, n in COMP_FILES[comp]:
            analyses[n], new_bytes[n] = analyse_file(os.path.join(p["raw"], n), kind)
    comps = {}
    for comp in COMPS:
        names = [n for _p, _k, n in COMP_FILES[comp]]
        v = t1s[comp].get("verdict")
        n_edit = sum(analyses[n]["counts"].get("root_props", 0) for n in names)
        probs = [analyses[n]["file"] + ": " + analyses[n]["scan_problem"] for n in names if analyses[n]["scan_problem"]]
        if v == "FAIL_STRUCT":
            d, why = "not_applicable", "T1 verdict FAIL_STRUCT: stop and report (PREREG_02 9.6)"
        elif n_edit == 0:
            d, why = "not_needed", "no game with removable root properties" + (f" (scan problems: {probs})" if probs else "")
        else:
            d, why = "candidate", f"{n_edit} games with removable root properties" + (
                f"; files not edited: {probs}" if probs else "")
        comps[comp] = {"t1_verdict": v, "decision": d, "reason": why, "t1": tp.rel_or_abs(t1_paths[comp], args.root),
                       "t1_sha256": tc.sha256_file(t1_paths[comp])}
    if os.path.exists(p["original"]):
        have = {n: (tc.sha256_file(os.path.join(p["original"], n)) if os.path.isfile(os.path.join(p["original"], n))
                    else None) for n in raw_sha}
        if have != raw_sha:
            raise Refuse(f"{p['original']} holds other files than data/test/raw: keep it, move it aside by hand and "
                         f"report to the main session")
    for sub in ("normalised", "check_rank", "check_player"):
        d = os.path.join(p["rp"], sub)
        if os.path.exists(d):
            tp.make_writable_tree(d)
            shutil.rmtree(d)
    for comp in COMPS:
        if comps[comp]["decision"] != "candidate":
            continue
        before = t1_profile(tc.t1_check(comp, p["raw"]))
        if before["verdict"] != t1s[comp].get("verdict"):
            raise Refuse(f"t1_check on the raw {comp} files gives {before['verdict']}, the committed t1.json "
                         f"{t1s[comp].get('verdict')}")
        chk = os.path.join(p["rp"], "check_" + comp)
        for _pool, _kind, n in COMP_FILES[comp]:
            tp.write_bytes(os.path.join(chk, n), new_bytes[n] if new_bytes[n] is not None else
                           tp.read_bytes(os.path.join(p["raw"], n)))
        try:
            after = t1_profile(tc.t1_check(comp, chk))
        finally:
            shutil.rmtree(chk, ignore_errors=True)
        comps[comp]["t1_profile_before"] = before
        comps[comp]["copy_t1_predicted"] = after
        if after == before:
            comps[comp]["decision"] = "apply"
        else:
            comps[comp]["decision"] = "not_applicable"
            comps[comp]["reason"] += "; the copy's T1 differs from the raw T1 (rev 11 section 3 item 4): stop and report"
    if os.path.isdir(p["rp"]) and not os.listdir(p["rp"]):
        os.rmdir(p["rp"])
    applied = [c for c in COMPS if comps[c]["decision"] == "apply"]
    files = {}
    for comp in COMPS:
        for pool, kind, n in COMP_FILES[comp]:
            a = analyses[n]
            edit = comp in applied and new_bytes[n] is not None
            after = new_bytes[n] if edit else None
            f = {k: v for k, v in a.items() if k != "edited"}
            f.update(comp=comp, pool=pool, kind=kind, edited=edit, edited_examples=a["edited"],
                     sha256_after=tp.sha256_bytes(after) if edit else a["sha256_before"],
                     bytes_after=len(after) if edit else a["bytes_before"])
            files[n] = f
    rec = {"format": FORMAT_PREPARE, "registration": REGISTRATION, "created": tp.now_iso(),
           "mode": "real" if real else "scratch", "root": None if real else os.path.abspath(args.root),
           "decision": "apply" if applied else "not_applicable_or_not_needed", "applied_comps": applied,
           "comps": comps, "files": files, "rev9": r9, "keep": sorted(KEEP),
           "raw": {"dir": tp.rel_or_abs(p["raw"], args.root), "sha256": raw_sha, "source_json": src.get("sources")},
           "original_dir": tp.rel_or_abs(p["original"], args.root) if applied else None,
           "normalised_dir": tp.rel_or_abs(p["normalised"], args.root) if applied else None,
           "rerun_reason": args.reason if replaced else None,
           "tool": {"file": "tools/root_props.py", "sha256": tc.sha256_file(os.path.abspath(__file__)),
                    "t02_checks_sha256": tc.sha256_file(os.path.join(HERE, "t02_checks.py")),
                    "trailing_pass_sha256": tc.sha256_file(os.path.join(HERE, "trailing_pass.py")),
                    "git_head": tc.git_head()}}
    os.makedirs(p["rec"], exist_ok=True)
    for comp in COMPS:
        tp.write_bytes(os.path.join(p["rec"], f"pre_t1_{comp}.json"), tp.read_bytes(t1_paths[comp]))
    if applied:
        for n in ALL_NAMES + ["SOURCE.json"]:
            o = os.path.join(p["original"], n)
            if not os.path.isfile(o):
                tp.write_bytes(o, tp.read_bytes(os.path.join(p["raw"], n)))
                os.chmod(o, stat.S_IREAD)
        for n in ALL_NAMES:
            data = new_bytes[n] if files[n]["edited"] else tp.read_bytes(os.path.join(p["raw"], n))
            tp.write_bytes(os.path.join(p["normalised"], n), data)
            if tp.sha256_bytes(data) != files[n]["sha256_after"]:
                raise Refuse(f"{n}: copy SHA-256 mismatch")
        orig = {n: tc.sha256_file(os.path.join(p["original"], n)) for n in ALL_NAMES + ["SOURCE.json"]}
        if orig != raw_sha:
            raise Refuse("the original copies differ from data/test/raw")
    tc.write_json(out_json, rec)
    for comp in COMPS:
        c = comps[comp]
        print(f"{comp}: T1 {c['t1_verdict']} -> {c['decision']} ({c['reason']})")
    for n in ALL_NAMES:
        f = files[n]
        print(f"  {n}: games {f['games']}, edited {f['games_edited'] if f['edited'] else 0}, counts {f['counts']}, "
              f"removed {f['removed_props']}, HD {f['hd_values']}, plain form {f['plain_form_before']} -> "
              f"{f['plain_form_after'] if f['edited'] else f['plain_form_before']}"
              + (f", SCAN PROBLEM {f['scan_problem']}" if f["scan_problem"] else ""))
    for n in ALL_NAMES:
        f = files[n]
        left = f["games"] - (f["plain_form_after"] if f["edited"] else f["plain_form_before"])
        if (f["edited"] or f["games_edited"]) and left:
            print(f"  WARNING {n}: {left} games are still not in the plain '(;SZ[19];B..;W..)' form (KataGo cannot "
                  f"read them): report the counts to the main session")
    print("wrote", out_json)
    stop = [c for c in COMPS if comps[c]["decision"] == "not_applicable"]
    if stop:
        print(f"STOP: {stop} not applicable ({'; '.join(comps[c]['reason'] for c in stop)}): report to the main "
              f"session; do not ingest" + (" (the copy was written for the record only)" if applied else ""))
        return 3
    if not applied:
        print("rev 11 is not needed: go on with docs/TESTDAY.md (nothing changed)")
        return 2
    if real:
        print("next (docs/TESTDAY_22_ROOTPROPS.md): commit results/test02/checks/root_props, then")
        print(f'  PY tools/t02_test_bundle.py ingest --src data/test_rootprops/normalised --rerun --reason "{REASON}"')
        for comp in COMPS:
            print(f'  PY tools/t02_checks.py t1 --comp {comp} --rerun --reason "{REASON}"')
        print("  commit results/test02/checks/{rank,player}, then  PY tools/root_props.py verify")
    return 0


# ================================================================ verify
def sha_or_none(path):
    return tc.sha256_file(path) if os.path.isfile(path) else None


def cmd_verify(args):
    real = args.root is None
    tp.check_root(args.root)
    p = paths(args.root)
    guard_paths(p)
    pj = os.path.join(p["rec"], "prepare.json")
    if not os.path.isfile(pj):
        raise Refuse(f"{pj} missing: run prepare first")
    if real:
        committed_clean([os.path.abspath(__file__), pj, os.path.join(REPO, *REGISTRATION.split("/")),
                         os.path.join(HERE, "t02_checks.py"), os.path.join(HERE, "trailing_pass.py")]
                        + [os.path.join(p["rec"], f"pre_t1_{c}.json") for c in COMPS],
                        "the rev 11 tool, registration, t02_checks.py, trailing_pass.py and the prepare records")
    prep = tc.read_json(pj)
    if prep.get("format") != FORMAT_PREPARE or not prep.get("applied_comps"):
        raise Refuse(f"{pj}: no competition was applied; nothing to verify")
    if any(prep["comps"][c]["decision"] == "not_applicable" for c in COMPS):
        raise Refuse(f"{pj}: a competition is not applicable (prepare said STOP); report to the main session")
    out = os.path.join(p["rec"], "verify.json")
    replaced = tp.require_fresh([out], args)
    probs, checks = [], {}
    files = prep["files"]
    for k, f in (("sha256", os.path.abspath(__file__)), ("t02_checks_sha256", os.path.join(HERE, "t02_checks.py")),
                 ("trailing_pass_sha256", os.path.join(HERE, "trailing_pass.py"))):
        if prep["tool"].get(k) != tc.sha256_file(f):
            probs.append(f"{os.path.basename(f)} changed since prepare")
    if os.path.exists(p["manifest"]) or os.path.exists(p["built"]):
        probs.append("the bundle was already built (data/test/MANIFEST.json or results/test02/checks/built exists): "
                     "STOP and report to the main session")
    for n in ALL_NAMES:
        f = files[n]
        got = {k: sha_or_none(os.path.join(p[k], n)) for k in ("original", "normalised", "raw")}
        checks[n] = got
        if got["original"] != f["sha256_before"]:
            probs.append(f"{n}: the original copy changed")
        if got["normalised"] != f["sha256_after"]:
            probs.append(f"{n}: the normalised copy changed")
        if got["raw"] != f["sha256_after"]:
            probs.append(f"{n}: data/test/raw is not the rev 11 copy (ingest --src data/test_rootprops/normalised "
                         f"--rerun)")
    for comp in COMPS:
        for _pool, kind, n in COMP_FILES[comp]:
            f = files[n]
            if checks[n]["original"] is None or checks[n]["raw"] is None:
                continue
            _a, nb = analyse_file(os.path.join(p["original"], n), kind)
            want = tp.sha256_bytes(nb) if (nb is not None and f["edited"]) else f["sha256_before"]
            if want != f["sha256_after"]:
                probs.append(f"{n}: re-deriving the copy from the original gives {want[:12]}, prepare.json "
                             f"{f['sha256_after'][:12]}")
            if comp in prep["applied_comps"]:
                again, nb2 = analyse_file(os.path.join(p["raw"], n), kind)
                if nb2 is not None or again["counts"].get("root_props"):
                    probs.append(f"{n}: data/test/raw still has removable root properties (not idempotent)")
    if sha_or_none(os.path.join(p["original"], "SOURCE.json")) != prep["raw"]["sha256"]["SOURCE.json"]:
        probs.append("the original SOURCE.json changed")
    src = tc.read_json(os.path.join(p["raw"], "SOURCE.json"))
    srcs = src.get("sources") or []
    if src.get("format") != SOURCE_FORMAT or len(srcs) != 1 or not tc.same_path(srcs[0], p["normalised"]):
        probs.append(f"data/test/raw/SOURCE.json sources {srcs}: not the single rev 11 copy {p['normalised']}")
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
        rs = tp.recorded_shas(t1)
        c = prep["comps"][comp]
        prof = t1_profile(t1)
        pre = os.path.join(p["rec"], f"pre_t1_{comp}.json")
        if sha_or_none(pre) != c["t1_sha256"]:
            probs.append(f"{pre} is not the T1 that prepare read")
        elif c["decision"] == "apply" and prof != t1_profile(tc.read_json(pre)):
            probs.append(f"{comp}: the T1 of the copy differs from the T1 before rev 11")
        if c["decision"] == "apply" and "rev 11" not in str((t1.get("provenance") or {}).get("rerun_reason")):
            probs.append(f"{comp}: t1.json was not rerun with a 'PREREG_02 rev 11' reason")
        t1rep[comp] = {"verdict": t1.get("verdict"), "sha256": tc.sha256_file(path),
                       "rerun_reason": (t1.get("provenance") or {}).get("rerun_reason")}
        for _pool, _kind, n in COMP_FILES[comp]:
            if rs.get(n) != files[n]["sha256_after"]:
                probs.append(f"{comp} t1.json records {str(rs.get(n))[:12]} for {n}, not the copy: rerun t1 --comp "
                             f"{comp} --rerun --reason ...")
        if c["decision"] == "apply":
            want = c["copy_t1_predicted"]
            for k in ("verdict", "normalize", "subchecks", "T1.4", "T1.5"):
                if prof.get(k) != want.get(k):
                    probs.append(f"{comp}: T1 {k} of the copy is {prof.get(k)}, predicted {want.get(k)}")
        elif t1.get("verdict") != c["t1_verdict"]:
            probs.append(f"{comp}: T1 verdict {t1.get('verdict')} differs from before ({c['t1_verdict']})")
    passed = not probs
    rec = {"format": FORMAT_VERIFY, "registration": REGISTRATION, "created": tp.now_iso(),
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
        print("next: commit results/test02/checks (root_props/verify.json and both t1.json), then docs/TESTDAY.md "
              "step 2 (build)")
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
