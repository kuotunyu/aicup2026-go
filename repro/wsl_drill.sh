#!/bin/bash
# Local Linux drill of the layer-1 reproduction (WSL 2), the same steps as repro/colab_layer1.ipynb:
#   clone data/delivery/C10_code.bundle (core.autocrlf=false), unzip the two feature zips into the tree, build a
#   Python 3.12.13 environment with uv from repro/requirements_cpu.txt (without colorama), then run each entry point
#   without a flag (does it stop at the same seam as on Colab?) and with --tolerant.
#
# Run from Windows (Git Bash), the script read through `tr` because the Windows working copy may be CRLF:
#   MSYS_NO_PATHCONV=1 wsl.exe -d <distro> -- bash -lc "tr -d '\r' < /mnt/d/AI-Portfolio/competitions/aicup2026-go/repro/wsl_drill.sh | bash -s -- all"
# Steps: setup | run | all.  Work folder: ~/c10_repro (tree c10/, venv312/, out/).  Nothing is written to /mnt.
# The host's free memory is not visible from WSL: check it on the host before starting and `wsl --shutdown` afterwards.
# data/local is not in the delivery package; the script stops if the tree has it.
set -euo pipefail
STEP=${1:-all}
REPO_WIN=${C10_REPO_WIN:-/mnt/d/AI-Portfolio/competitions/aicup2026-go}
D=$REPO_WIN/data/delivery
W=$HOME/c10_repro
export PATH=$HOME/.local/bin:$PATH

setup() {
  rm -rf "$W"; mkdir -p "$W"
  git -c core.autocrlf=false clone -q "$D/C10_code.bundle" "$W/c10"
  git -C "$W/c10" config core.autocrlf false
  echo "commit $(git -C "$W/c10" rev-parse HEAD)"
  cd "$W/c10"
  for z in C10_features_layer1_comp2.zip C10_features_layer1_comp1.zip; do
    python3 -c '
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    names = [i.filename for i in z.infolist()]
    bad = [x for x in names if x.startswith(("/", "..")) or "/../" in x or not x.startswith("data/") or x.lower().startswith("data/local")]
    assert not bad, bad[:5]
    z.extractall(".")
    print(sys.argv[1], len(names), "files")
' "$D/$z"
  done
  test ! -e data/local
  test -z "$(git status --porcelain --untracked-files=no)"
  uv python install 3.12.13
  uv venv "$W/venv312" --python 3.12.13
  grep -v '^#' repro/requirements_cpu.txt | grep -vi '^colorama' | grep . > "$W/req.txt"
  uv pip install --python "$W/venv312/bin/python" -r "$W/req.txt" \
     --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
   || uv pip install --python "$W/venv312/bin/python" $(grep -iE '^(numpy|scipy|psutil)==' "$W/req.txt")
  "$W/venv312/bin/python" -c "import sys, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"
  echo SETUP_DONE
}

run() {
  cd "$W/c10"
  unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS PYTHONPATH PYTHONHOME
  export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
  mkdir -p "$W/out"
  TIME=""; if [ -x /usr/bin/time ]; then TIME="/usr/bin/time -v"; fi   # peak RSS of each run, when available
  for comp in comp2 comp1; do
    for mode in bytewise tolerant; do
      tag=$comp; flag=""
      if [ $mode = tolerant ]; then tag=${comp}_tolerant; flag="--tolerant"; fi
      rm -rf "$W/out/$tag"
      echo "########## $comp $mode  $(date +%T)"
      set +e
      $TIME "$W/venv312/bin/python" -u repro/${comp}_from_features.py --root "$W/c10" --out "$W/out/$tag" $flag \
          > "$W/out/${tag}_log.txt" 2>&1
      rc=$?
      set -e
      echo "exit code $rc"
      grep -E '^(DRIFT|REFUSED|FAIL|WATCHDOG|seam |\[tolerant\]|new questions|  sha256|  expected|  PASS|  FAIL|  rows different|    row |seconds )|differs|Maximum resident' \
          "$W/out/${tag}_log.txt" | cut -c1-400 || true
    done
  done
  test ! -e data/local
  echo "git status (tracked): [$(git status --porcelain --untracked-files=no)]"
  echo RUN_DONE
}

case "$STEP" in
  setup) setup ;;
  run) run ;;
  all) setup; run ;;
  *) echo "usage: wsl_drill.sh setup|run|all"; exit 2 ;;
esac
