"""prereg25 exploration (competition 2): read-only audit of the validation players and the V sets (CPU, .venv).

    Git Bash, repository root (subst drive in the worktree); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe
    S=results/prereg25/explore
    $S/p25_data.py audit   -> $S/audit.json   (counts only: no model, no score, no feature array)
    $S/p25_data.py sets    -> data/prereg25/v/sets.json (p1_extract sets layout, probe source 'val'; the game list
                              must have the SHA-256 that audit.json recorded)

Validation players come only from the committed split v1 file (split == 'val'); their games and levels only from the
official training CSVs. Inner / fit games through tools/c10data.load_rows (local_dir None). Never data/local, never
tools/score.py.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p25_lib as L  # noqa: E402
import c10data  # noqa: E402

REPO = L.REPO
SETS_P = {"p1_probe": os.path.join(REPO, "data", "prereg08", "p1", "sets.json"),
          "p2_probe": os.path.join(REPO, "data", "prereg08", "p2", "sets.json"),
          "p3_probe": os.path.join(REPO, "data", "prereg09", "p3", "sets.json")}
P1_SETS = SETS_P["p1_probe"]
RUNLOGS = {"p3": os.path.join(REPO, "data", "prereg09", "p3", "run_log.jsonl"),
           "p2": os.path.join(REPO, "data", "prereg08", "p2", "run_log.jsonl")}
AUDIT = os.path.join(HERE, "audit.json")
V_DIR = os.path.join(REPO, "data", "prereg25", "v")
V_SETS = os.path.join(V_DIR, "sets.json")


def log(*x):
    print(*x, flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def sha_raw(path):
    with open(L.guard(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def games_sha(games):
    return hashlib.sha256(json.dumps(games, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_end(path):
    d = None
    with open(L.guard(path), encoding="utf-8") as f:
        for line in f:
            if '"run end"' in line:
                d = json.loads(line)
    return {"positions": d["positions"], "seconds": d["seconds"], "positions_per_s": d["positions"] / d["seconds"]}


def probe_games():
    out = {}
    for s, p in SETS_P.items():
        with open(L.guard(p), encoding="utf-8") as f:
            out[s] = json.load(f)["probe"]["games"]
    return out


def depth(games, levels=L.LEVELS):
    """games per player, per level: min / median / p90 / max and players."""
    c = Counter((g["rank"], g["player_id"]) for g in games)
    out = {}
    for lv in levels:
        v = np.array([n for (rk, _p), n in c.items() if rk == lv])
        out[lv] = ({"players": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                    "max": int(v.max()), "mean": float(v.mean())} if len(v) else {"players": 0})
    return out


def eligible(games, n):
    """players per level x colour with >= n games of that colour (the S19 sampler unit)."""
    c = Counter((g["rank"], g["player_id"], g["color"]) for g in games)
    out = {lv: [0, 0] for lv in L.LEVELS}
    for (rk, _p, col), k in c.items():
        if k >= n:
            out[rk][0 if col == "B" else 1] += 1
    return out


def select(log=log):
    """The deterministic V selection (PLAN.md 2.2) and every count the audit records."""
    t0 = time.time()
    vrows = L.load_val_rows()
    fit = c10data.load_rows(L.TRAIN_DIR, L.SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    inner = c10data.load_rows(L.TRAIN_DIR, L.SPLIT_CSV, part="inner", local_dir=None, check_counts=True)
    log(f"rows: val {len(vrows)}, fit {len(fit)}, inner {len(inner)} ({time.time() - t0:.0f}s)")
    split = c10data.read_split(L.SPLIT_CSV)
    vp = L.val_players()
    inner_p = {r["player_id"] for r in inner}
    fit_p = {r["player_id"] for r in fit}
    # ---- duplicates: a validation game equal to an inner game under any of the 8 symmetries is excluded
    inner_sym = set()
    for r in inner:
        inner_sym.update(L.sym_keys(r["sgf_content"]))
    log(f"inner symmetric keys {len(inner_sym)} ({time.time() - t0:.0f}s)")
    fit_key = {L.move_key(r["sgf_content"])[0] for r in fit}
    pg = probe_games()
    probe_set = {(g["rank"], g["game_id"]) for gs in pg.values() for g in gs}
    probe_key = {L.move_key(r["sgf_content"])[0] for r in fit if (r["rank"], r["game_id"]) in probe_set}
    vkey, vmoves = {}, {}
    dup_inner, dup_fit, dup_probe = [], 0, 0
    within = Counter()
    for r in vrows:
        k, nm = L.move_key(r["sgf_content"])
        g = (r["rank"], r["game_id"])
        vkey[g], vmoves[g] = k, nm
        within[k] += 1
        if k in inner_sym:
            dup_inner.append(g)
        dup_fit += k in fit_key
        dup_probe += k in probe_key
    dup_inner_set = set(dup_inner)
    usable = [r for r in vrows if (r["rank"], r["game_id"]) not in dup_inner_set]
    log(f"duplicates: val vs inner (8 symmetries) {len(dup_inner)}, vs fit {dup_fit}, vs probe {dup_probe} "
        f"({time.time() - t0:.0f}s)")
    # ---- quotas and the V sets
    fit_pl = {lv: len({r["player_id"] for r in fit if r["rank"] == lv}) for lv in L.LEVELS}
    val_pl = {lv: sum(1 for p, rk in vp.items() if rk == lv) for lv in L.LEVELS}
    quotas = L.scaled_quotas(val_pl, fit_pl)
    vsets = L.build_v_sets(usable, quotas)
    allv = [g for b, _q in L.PROBE_BATCHES for g in vsets[b]]
    if len({(g["rank"], g["game_id"]) for g in allv}) != len(allv):
        raise SystemExit("V sets: a game twice")
    if {g["player_id"] for g in allv} & (inner_p | fit_p):
        raise SystemExit("V sets: a training player")
    pos = {b: int(sum(vmoves[(g["rank"], g["game_id"])] for g in vsets[b])) for b in vsets}
    thr = {k: run_end(p) for k, p in RUNLOGS.items()}
    tot = sum(pos.values())
    probe_all = [g for gs in pg.values() for g in gs]
    per_level = {}
    for lv in L.LEVELS:
        lr = [r for r in vrows if r["rank"] == lv]
        per_level[lv] = {"val_players": val_pl[lv], "fit_players": fit_pl[lv],
                         "val_games": len(lr), "val_games_B": sum(r["color"] == "B" for r in lr),
                         "val_games_W": sum(r["color"] == "W" for r in lr),
                         "val_moves": int(sum(vmoves[(r["rank"], r["game_id"])] for r in lr)),
                         "quota": {b: quotas[b][lv] for b, _q in L.PROBE_BATCHES},
                         "selected": sum(1 for g in allv if g["rank"] == lv),
                         "selected_players": len({g["player_id"] for g in allv if g["rank"] == lv}),
                         "probe_games": sum(1 for g in probe_all if g["rank"] == lv)}
    vc = Counter(r["player_id"] for r in vrows)
    vv = np.array(list(vc.values()))
    rec = {
        "split_v1_sha256": sha_raw(L.SPLIT_CSV),
        "split_v1_players_by_split": dict(Counter(v["split"] for v in split.values())),
        "val": {"players": len(vp), "games": len(vrows), "moves": int(sum(vmoves.values())),
                "players_in_inner": len(set(vp) & inner_p), "players_in_fit": len(set(vp) & fit_p),
                "games_per_player": {"min": int(vv.min()), "median": float(np.median(vv)),
                                     "p90": float(np.percentile(vv, 90)), "max": int(vv.max())}},
        "duplicates": {"val_vs_inner_8_symmetries": len(dup_inner), "val_vs_inner_examples": dup_inner[:10],
                       "val_vs_fit_original": dup_fit, "val_vs_probe_original": dup_probe,
                       "within_val_original": int(sum(c - 1 for c in within.values() if c > 1)),
                       "rule": "validation games equal to an inner game under any of the 8 symmetries are excluded "
                               "before sampling; the others are record only"},
        "per_level": per_level,
        "v_sets": {"batches": {b: {"games": len(vsets[b]), "positions": pos[b], "seed": L.V_SEEDS[b],
                                   "players": len({g["player_id"] for g in vsets[b]})} for b in vsets},
                   "games": len(allv), "players": len({g["player_id"] for g in allv}), "positions": tot,
                   "games_sha256": games_sha(allv)},
        "depth_games_per_player": {"probe": depth(probe_all), "v": depth(allv)},
        "eligible_n_level_colour": {str(n): {"probe": eligible(probe_all, n), "v": eligible(allv, n)}
                                    for n in (5, 10, 15, 20)},
        "gpu_estimate": {"throughput_runs": thr,
                         "hours_4090": {k: tot / v["positions_per_s"] / 3600.0 for k, v in thr.items()},
                         "note": "positions = moves (one P1 position per move); same p1_extract.py run, fp16, "
                                 "batch 1024"},
        "seconds": round(time.time() - t0, 1),
    }
    return rec, vsets


def cmd_audit(_a):
    rec, _v = select()
    rec = {"format": "c10-prereg25 explore audit v1",
           "what": "read-only counts: validation players of split v1 (players_split.csv split == 'val') and their games "
                   "in the official training CSVs; the V selection (PLAN.md 2.2); no model, no score, no data/local",
           **rec}
    write_json(AUDIT, rec)
    log(json.dumps({k: rec[k] for k in ("val", "duplicates", "v_sets", "gpu_estimate")}, indent=1))
    return 0


def cmd_sets(_a):
    if os.path.exists(V_SETS):
        raise SystemExit(f"{V_SETS} exists; not overwritten")
    with open(AUDIT, encoding="utf-8") as f:
        want = json.load(f)["v_sets"]["games_sha256"]
    rec, vsets = select()
    allv = [g for b, _q in L.PROBE_BATCHES for g in vsets[b]]
    if games_sha(allv) != want:
        raise SystemExit("the V selection differs from audit.json")
    with open(L.guard(P1_SETS), encoding="utf-8") as f:
        s1 = json.load(f)
    out = {"format": L.SETS_FORMAT, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "probe": {"source": L.SOURCE, "seeds": L.V_SEEDS, "batches": {b: len(vsets[b]) for b in vsets},
                     "games": allv, "games_sha256": want,
                     "what": "prereg25 V: validation players of split v1, P1 / P2 / P3 round robin with quotas scaled "
                             "per level (PLAN.md 2.2); bank / mono copied from P1 and never run here"},
           "bank": s1["bank"], "mono": s1["mono"],
           "inputs_sha256": {"players_split.csv": sha_raw(L.SPLIT_CSV), "p1_sets.json": sha_raw(P1_SETS)},
           "counts": {"probe": len(allv), "probe_players": len({g["player_id"] for g in allv}),
                      "bank": s1["counts"]["bank"], "mono": s1["counts"]["mono"]}}
    write_json(V_SETS, out)
    log(json.dumps({"written": V_SETS, "sha256": sha_raw(V_SETS), "counts": out["counts"]}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    sub.add_parser("sets")
    a = ap.parse_args(argv)
    return {"audit": cmd_audit, "sets": cmd_sets}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
