#!/bin/bash
# E347 scratch: run a project script / test with the staged copies first (see stage_launch.py):   wsl bash temp/e347/run_staged.sh <script.py | basename> [args ...]
# the environment is the one of tools/env/_py14.sh (gcar env, bflc_env.sh PYTHONPATH); 2 threads, nice. BFLC_STAGE=none runs the real files (control).
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BFLC_ROOT="$R"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
source "$R/tools/env/bflc_env.sh"
exec nice -n 10 python "$R/temp/e347/stage_launch.py" "$@"
