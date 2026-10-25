#!/bin/bash
# smoke test of the designs given as arguments (20 iterations + VAL250 evaluation each)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gcar
source /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/bflc_env.sh
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
for d in "$@"; do
  echo "=== $d"
  python p16_train.py --design $d --name smoke_$d --smoke --eval val250 --workers 3 2>&1 | tail -8
done
