#!/bin/bash
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
rm -f $O/e116c.done
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar"
python p13_phead_fail.py --out e113_a --fw 3 > $O/e113_a/run_phead_fail.log 2>&1
echo done > $O/e116c.done
