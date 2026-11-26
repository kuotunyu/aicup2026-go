#!/usr/bin/env bash
# prereg42: D-player vectors of a finished variant (raw and EMA; V3 / V3b also full / chunk24), PLAN.md 4.
#   bash results/prereg42/explore/run_embed_D.sh V2 [step_40000] [--tta 8 --weights ema ...]
# Extra arguments after the step replace the default loop (one call of p42_embed.py with them).
# Log appended to data/prereg42/embed_log.txt; every output is resumable (parts) and skipped when it exists.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${C10_GPU_PY:-/d/AI-Portfolio/competitions/aicup2026-go/.venv-gpu/Scripts/python.exe}"
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
cd "$REPO" || exit 2
VAR="$1"; STEP="${2:-step_40000}"; shift; [ $# -gt 0 ] && shift
CK="data/prereg42/$VAR/ckpt/$STEP.pt"
[ "$VAR" = "base" ] && CK="${C10_BASE_CKPT:-data/prereg36/ckpt/step_40000.pt}"
LOG=data/prereg42/embed_log.txt
mkdir -p data/prereg42
run() { echo "=== $(date '+%F %T') embed $CK $*" >> "$LOG"; "$PY" -u results/prereg42/explore/p42_embed.py --ckpt "$CK" --players D --workers 4 "$@" >> "$LOG" 2>&1; local rc=$?; echo "=== $(date '+%F %T') rc=$rc" >> "$LOG"; return $rc; }
if [ $# -gt 0 ]; then run "$@"; exit $?; fi
if [ "$VAR" = "V3" ] || [ "$VAR" = "V3b" ]; then
  for w in raw ema; do for m in full chunk24; do run --weights $w --mode $m || exit $?; done; done
else
  for w in raw ema; do run --weights $w || exit $?; done
fi
