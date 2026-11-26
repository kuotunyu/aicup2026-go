"""Judge check (fit only, CPU, read-only): do the same games appear under two fit players?
If so, what is |level(target) - level(opponent)| ? Informs the colour-swap check and
how much target-perspective can add. Also per-player concentration (cap idea)."""
import sys, os, json, hashlib, re, time
sys.dont_write_bytecode = True
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
from c10data import load_rows, LEVELS
from collections import defaultdict, Counter
t0 = time.time()
rows = load_rows(os.path.join(REPO, "data", "training"), os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                 part="fit", local_dir=os.path.join(REPO, "data", "local"))
print("rows", len(rows), time.time() - t0, flush=True)
print("example head:", rows[0]["sgf_content"][:160], rows[0]["color"])
mv = re.compile(r";([BW])\[([a-s]{0,2})\]")
by = defaultdict(list)
for r in rows:
    moves = "".join(c + p for c, p in mv.findall(r["sgf_content"]))
    h = hashlib.sha1(moves.encode()).hexdigest()
    by[h].append((LEVELS.index(r["rank"]), r["color"], r["player_id"]))
dups = [v for v in by.values() if len(v) > 1]
print("unique games", len(by), "dup groups", len(dups), flush=True)
diff = Counter(); samecol = 0; opp = 0; samepl = 0
for v in dups:
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            a, b = v[i], v[j]
            if a[2] == b[2]:
                samepl += 1; continue
            if a[1] == b[1]:
                samecol += 1; continue
            opp += 1
            diff[abs(a[0] - b[0])] += 1
out = {"rows": len(rows), "unique_games": len(by), "dup_groups": len(dups), "pairs_opposite_colour": opp,
       "pairs_same_colour_diff_player": samecol, "pairs_same_player": samepl,
       "abs_level_diff_opposite": dict(sorted(diff.items()))}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dup_opp.json"), "w"), indent=1)
