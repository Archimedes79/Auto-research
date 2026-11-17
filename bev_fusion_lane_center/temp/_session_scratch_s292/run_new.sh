#!/bin/bash
# run a scratch copy against the live tree: run_new.sh <script in new/> [args]   (cwd = the live folder of the module, scratch dir FIRST on PYTHONPATH)
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BFLC_ROOT="$R"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar"
source $R/tools/env/bflc_env.sh
export PYTHONPATH="$R/temp/_session_scratch_s292/new:$PYTHONPATH"
cd "$R/${RUN_CWD:-training/grid}"
python "$@"
