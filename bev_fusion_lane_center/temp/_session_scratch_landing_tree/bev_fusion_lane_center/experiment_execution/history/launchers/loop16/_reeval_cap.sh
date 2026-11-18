#!/bin/bash
# capacity re-evaluation (inference only): beam M = 192, seeds K0 = 96 for finished runs given as "<design>:<run name>" pairs; card tag m192
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gcar
source /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/bflc_env.sh
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
for pair in "$@"; do
  d=${pair%%:*}; n=${pair##*:}
  python p16_eval.py --design $d --name $n --tag m192 --set M=192 K0=96 2>&1 | head -1
done
