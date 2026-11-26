#!/usr/bin/env bash
# PREREG_10 rev 4 (part B): detached 4090 launcher for tools/t10_player.py, so a long GPU step is not tied to a chat.
#   WHAT = rehearse   p1-rehearse   (before 2026-11-04 11:00; ~2 min; then commit results/test10/p1/p1_rehearsal.json)
#          seam_val   p1 --pool seam_val   (the GPU seam; ~2 min)
#          public     p1 --pool public     (~2.7 h)
#          private    p1 --pool private    (~2.7 h)
# Only after B is adopted; otherwise t10_player.py exits 3 (B not adopted) and this launcher ends.
# Launch (PowerShell, repository root D:\AI-Portfolio\competitions\aicup2026-go):
#   Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/d/AI-Portfolio/competitions/aicup2026-go/tools/t10_p1_launch.sh public" -WindowStyle Hidden
# Log: data/test10/p1/launch_<WHAT>.log. Waits while another python compute process is on the GPU (every 5 min).
# Yield the GPU: create data/test10/p1/<pool>/STOP (data/test10/p1/rehearsal/STOP for the rehearsal): the step stops
# after the chunk in progress (exit 5) and the launcher ends; delete STOP and launch again to resume.
# Exit codes are written to the log: 0 done (commit the report), 2 refused (read the log), 3 B not used, 5 stopped.
cd /d/AI-Portfolio/competitions/aicup2026-go || exit 1
WHAT="$1"
case "$WHAT" in
  rehearse) ARGS=(p1-rehearse) ;;
  seam_val|public|private) ARGS=(p1 --pool "$WHAT") ;;
  *) echo "usage: t10_p1_launch.sh rehearse|seam_val|public|private" >&2; exit 2 ;;
esac
mkdir -p data/test10/p1
LOG="data/test10/p1/launch_${WHAT}.log"
export PYTHONDONTWRITEBYTECODE=1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS PYTHONUTF8
echo "launcher start $(date) ${ARGS[*]}" >> "$LOG"
while true; do
  busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -ci "python")
  if [ "$busy" -gt 0 ]; then echo "GPU busy ($busy python compute process), waiting 5 min $(date)" >> "$LOG"; sleep 300; continue; fi
  .venv-gpu/Scripts/python.exe tools/t10_player.py "${ARGS[@]}" >> "$LOG" 2>&1
  rc=$?
  echo "t10_player ${ARGS[*]} rc=$rc $(date)" >> "$LOG"
  break
done
echo "launcher end $(date)" >> "$LOG"
