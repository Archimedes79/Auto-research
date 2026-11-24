#!/bin/bash
# E358 scratch: run a project script with the STAGED copies of qends_eval.py / jends_eval.py / jwire_head.py first on the PYTHONPATH (the originals are in E343's code digest and stay untouched):
#   wsl bash temp/e358/run_staged.sh jends_eval.py score --run e324_s_s0 --seed 0 --thr 0.1 --attach --snap --snap-nogeom --wire twin --tag twinatt
# the environment is the one of tools/env/_py14.sh (gcar env, bflc_env.sh PYTHONPATH); CPU only, 2 threads, nice.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BFLC_ROOT="$R"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
source "$R/tools/env/bflc_env.sh"
S="$R/temp/_session_scratch_e358/stage"
export PYTHONPATH="$S:$PYTHONPATH"
s="$1"; shift
[ -f "$S/$s" ] || { echo "run_staged: $s is not a staged file"; exit 2; }
cd "$S"
exec nice -n 10 python "$s" "$@"
