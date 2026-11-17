#!/bin/bash
# a variant of _py14.sh for scratch timing: no PYTHONDONTWRITEBYTECODE
R=/mnt/c/Auto-research/bev_fusion_lane_center
export BFLC_ROOT="$R"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar"
source $R/tools/env/bflc_env.sh
cd "$(dirname "$1")"
python "$(basename "$1")" "${@:2}"
