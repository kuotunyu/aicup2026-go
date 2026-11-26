"""U3 (prereg 05 exploration, timeline): small numbers behind the timeline note. Stdlib only, no data files.

1. Weekdays of the dates used in the critical path (2026).
2. Adoption probability of a single validation scoring (H5R) for a few true validation gains and SEs, with the
   normal approximation used by tools/p03_rank_fuse.se_val_record: P = Phi(delta / SE - 1.96).
   SE reference points (inner-based, written before any validation scoring, results/prereg04/rank/tune_report.json
   se_val_record): Q01 vs R20 0.00694, Q02 vs R20 0.00738. An increment on top of Q02 changes fewer decisions, so
   its SE should be smaller; 0.003-0.006 is an assumed range, not a measurement.
3. Elapsed hours of past cycles from git commit times (copied from `git log`, 2026-10-01).
Writes u3_timeline_numbers.json next to this file.
"""
import datetime as dt
import json
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "u3_timeline_numbers.json")
WD = "一二三四五六日"


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


dates = ["2026-10-01", "2026-10-04", "2026-10-05", "2026-10-07", "2026-10-09", "2026-10-12", "2026-10-14",
         "2026-10-16", "2026-10-18", "2026-10-19", "2026-10-24", "2026-10-27", "2026-10-30", "2026-10-31",
         "2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-07", "2026-11-10",
         "2026-11-15", "2026-11-16", "2026-11-20", "2026-11-23", "2026-11-24"]
weekdays = {d: WD[dt.date.fromisoformat(d).weekday()] for d in dates}

ses = [0.003, 0.004, 0.005, 0.006, 0.00738]
deltas = [0.003, 0.005, 0.008, 0.010, 0.015, 0.020]
adoption = {f"SE={s}": {f"delta={d}": round(phi(d / s - 1.96), 3) for d in deltas} for s in ses}
# smallest true gain with >= 50 % / >= 80 % adoption at each SE
need = {f"SE={s}": {"p50": round(1.96 * s, 4), "p80": round((1.96 + 0.8416) * s, 4)} for s in ses}

T = dt.datetime.fromisoformat
cycles = {
    "prereg02 rev1 -> impl (a7fe166)": (T("2026-09-26T05:18"), T("2026-09-26T07:34")),
    "prereg02 test-day kit rev5 -> e2e fixes (74486a2)": (T("2026-09-26T15:17"), T("2026-09-26T18:57")),
    "prereg03 rev1 -> impl with 3 reviews (39ebccb)": (T("2026-09-26T22:53"), T("2026-09-27T05:59")),
    "prereg03 Colab unpack -> RESULTS (e3fc40a)": (T("2026-09-29T01:57"), T("2026-09-29T02:11")),
    "prereg04 rev1 -> impl rev2 (82027fa)": (T("2026-09-29T12:22"), T("2026-09-29T16:23")),
    "prereg04 impl -> H4R scored (e8b24f2)": (T("2026-09-29T16:23"), T("2026-09-29T17:10")),
    "prereg04 rev1 -> RESULTS verified (608d702)": (T("2026-09-29T12:22"), T("2026-09-29T17:23")),
    "test04 branch point 6ec5e68 -> design 9f63acd": (T("2026-09-29T17:46"), T("2026-09-30T03:55")),
    "test04 design -> implementation 62ca557": (T("2026-09-30T03:55"), T("2026-09-30T17:35")),
    "test04 implementation -> merge f4853e5 (2 K5 gate rounds)": (T("2026-09-30T17:35"), T("2026-10-01T01:54")),
    "test04 branch point -> merge (whole test-day extension)": (T("2026-09-29T17:46"), T("2026-10-01T01:54")),
}
elapsed = {k: round((b - a).total_seconds() / 3600, 2) for k, (a, b) in cycles.items()}

out = {"weekdays": weekdays, "adoption_probability": adoption, "true_gain_needed": need,
       "formula": "Phi(delta / SE - 1.96) (tools/p03_rank_fuse.se_val_record)",
       "se_reference_inner_based": {"Q01_vs_R20": 0.00693594129100802, "Q02_vs_R20": 0.007378616638272913,
                                    "source": "results/prereg04/rank/tune_report.json combinations[1|2].oof"
                                              ".se_val_record (inner only, before H4R)"},
       "elapsed_hours_from_git": elapsed}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
