#!/bin/bash
# E347: run a project script with the STAGED copies of the frozen lane files in front of the real ones (the E343 code digest is frozen), on the GPU (nothing disabled, no nice: this is the wrapper of the run.py lines).
# STANDALONE (a copy of the generic launcher of temp/e291/run_staged_gpu.sh with the E347 stage as the default and BFLC_SIG=1.0): bash reads a script by offset, so an edit of another chat's launcher must not be able to break a running training.
#   bash temp/e347/run_staged_gpu.sh <script.py | basename> [args ...]        (cwd = the problem folder; run.py starts it)
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BFLC_ROOT="$R"
export BFLC_STAGE="${BFLC_STAGE:-$R/temp/_session_scratch_e347/stage}"
export BFLC_SIG="${BFLC_SIG:-1.0}"
[ "$BFLC_STAGE" = "none" ] || [ -f "$BFLC_STAGE/training/grid/jmaps.py" ] || { echo "run_staged_gpu.sh: no staged jmaps.py in $BFLC_STAGE: this run would use the REAL files" >&2; exit 2; }
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar"
source "$R/tools/env/bflc_env.sh"
exec python "$R/temp/e347/stage_launch.py" "$@"
