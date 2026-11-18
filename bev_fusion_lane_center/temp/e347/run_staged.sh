#!/bin/bash
# E347 scratch: run a project script / test with the staged copies first (see stage_launch.py):   wsl bash temp/e347/run_staged.sh <script.py | basename | path> [args ...]
# the environment is the one of tools/env/_py14.sh (gcar env, bflc_env.sh PYTHONPATH); 2 threads, nice. BFLC_STAGE=none runs the real files (control).
# E347_FAST=1: reuse the PYTHONPATH of the last full run (bflc_env.sh walks the whole tree over the slow /mnt/c mount, about 55 s); stale when a folder is added, so never for a final run.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BFLC_ROOT="$R"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""    # CPU only: nothing in a staged run may touch the GPU slots of the running jobs
CACHE="$R/temp/_session_scratch_e347/pythonpath.txt"
if [ "$E347_FAST" = "1" ] && [ -s "$CACHE" ]; then
  export PYTHONPATH="$(cat "$CACHE")"
else
  source "$R/tools/env/bflc_env.sh"
  echo "$PYTHONPATH" > "$CACHE"
fi
exec nice -n 10 python "$R/temp/e347/stage_launch.py" "$@"
