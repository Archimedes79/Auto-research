#!/bin/bash
# scratch runner: the gcar env WITHOUT the 319-entry PYTHONPATH of _py14.sh (it makes `import torch` take 25 s on this box); usage: np.sh <script path under the project> args
R=/mnt/c/Auto-research/bev_fusion_lane_center
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export BFLC_ROOT="$R" PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar" OMP_NUM_THREADS=${NP_THREADS:-2} MKL_NUM_THREADS=${NP_THREADS:-2}
unset PYTHONPATH
s="$1"; shift
cd "$(dirname "$R/$s")"
exec nice -n 10 python "$(basename "$s")" "$@"
