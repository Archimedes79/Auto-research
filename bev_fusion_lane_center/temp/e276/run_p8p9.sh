#!/bin/bash
# scratch: P8 (assignment 1:1) and P9 (link calibration, risk-coverage) on the full prediction dumps of B (seed 0, 1), CPU only, low priority (two GPU jobs train)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1; do
  nice -n 12 bash tools/env/_py14.sh exp_assignment_uncertainty.py /mnt/c/Auto-research/bev_fusion_lane_center/temp/e276/predfull_e273_p_s$s.pkl /mnt/c/Auto-research/data/autogt/lane/0702a065c6 --pictures 6 > temp/e276/p8p9_s$s.log 2>&1
done
touch temp/e276/P8P9_DONE
