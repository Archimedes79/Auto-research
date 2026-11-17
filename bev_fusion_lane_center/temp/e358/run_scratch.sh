#!/bin/bash
# E358 scratch: run a scratch script by path in the gcar env (CPU only, 2 threads, nice), the stage dir first on the PYTHONPATH:
#   wsl bash temp/e358/run_scratch.sh temp/_session_scratch_e358/test_staged.py [args]
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BFLC_ROOT="$R"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
source "$R/tools/env/bflc_env.sh"
export PYTHONPATH="$R/temp/_session_scratch_e358/stage:$PYTHONPATH"
s="$1"; shift
[ -f "$s" ] || s="$R/$s"
cd "$(dirname "$s")"
exec nice -n 10 python "$(basename "$s")" "$@"
